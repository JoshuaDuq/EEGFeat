package app

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JoshuaDuq/EEGFeat/tui/eegfeat"
)

func TestLongReviewKeepsEverySelectedRowVisible(t *testing.T) {
	for _, field := range []string{"exclude", "bads"} {
		t.Run(field, func(t *testing.T) {
			gate := icaGate()
			gate.Field = field
			gate.Items = nil
			for i := 0; i < 80; i++ {
				gate.Items = append(gate.Items, eegfeat.Item{ID: []byte(fmt.Sprint(i)), Label: fmt.Sprintf("item-%03d", i)})
			}
			m := openGate(t, &fakeBackend{status: statusFixture(), gate: gate})
			for _, height := range []int{24, 40, 24} {
				m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: height})
				for _, direction := range []tea.KeyType{tea.KeyDown, tea.KeyUp} {
					for i := 0; i < m.gate.rows(); i++ {
						view := m.View()
						label := "add"
						if m.gate.cursor < len(gate.Items) {
							label = gate.Items[m.gate.cursor].Label
						}
						assertFocusedRow(t, view, label)
						if lipgloss.Height(view) > height || lipgloss.Width(view) > 100 {
							t.Fatalf("view exceeds terminal size: %dx%d", lipgloss.Width(view), lipgloss.Height(view))
						}
						m, _ = update(t, m, special(direction))
					}
				}
			}
		})
	}
}

func assertFocusedRow(t *testing.T, view, label string) {
	t.Helper()
	for _, line := range strings.Split(view, "\n") {
		if strings.Contains(line, "▎") && strings.Contains(line, label) {
			return
		}
	}
	t.Fatalf("selected row %q is hidden:\n%s", label, view)
}

func TestLongRecordingListFollowsSelection(t *testing.T) {
	status := statusFixture()
	for i := 0; i < 80; i++ {
		status.Recordings = append(status.Recordings, eegfeat.Recording{Label: fmt.Sprintf("recording-%03d", i)})
	}
	m := home(t, &fakeBackend{status: status})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	for i := range status.Recordings {
		assertFocusedRow(t, m.View(), status.Recordings[i].Label)
		m, _ = update(t, m, special(tea.KeyDown))
	}
}

func TestReviewNavigationAndSortPreserveComponentIdentity(t *testing.T) {
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: icaGate()})
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, key('o'))
	assertFocusedRow(t, m.View(), "ICA001")
	m, _ = update(t, m, special(tea.KeyEnd))
	assertFocusedRow(t, m.View(), "ICA003")
	m, _ = update(t, m, special(tea.KeyHome))
	assertFocusedRow(t, m.View(), "ICA000")
	m, _ = update(t, m, special(tea.KeyPgDown))
	assertFocusedRow(t, m.View(), "ICA003")
	m, _ = update(t, m, special(tea.KeyPgUp))
	assertFocusedRow(t, m.View(), "ICA000")
}

func TestBusyOperationsIgnoreRepeatedActions(t *testing.T) {
	for _, busy := range []string{"loading", "starting", "opening", "saving", "resetting"} {
		t.Run(busy, func(t *testing.T) {
			m := home(t, &fakeBackend{status: statusFixture()})
			if busy == "saving" {
				gate := newGate("sub-01", icaGate())
				m.gate = &gate
			}
			m.busy = busy
			for _, msg := range []tea.KeyMsg{key('r'), key('a'), key('v'), special(tea.KeyEnter), special(tea.KeyEsc)} {
				next, cmd := update(t, m, msg)
				if cmd != nil || next.gate != m.gate || next.busy != busy {
					t.Fatalf("%s accepted %s while busy", busy, msg.String())
				}
			}
		})
	}
}

func TestRunCannotOpenReviewWhileWritingCheckpoints(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := startRun(t, backend, 40)
	_, cmd := update(t, m, special(tea.KeyEnter))
	if cmd != nil {
		t.Fatal("review opened during a run")
	}
}

func TestSlowRefreshCannotOverwriteANewerStatus(t *testing.T) {
	old := statusFixture()
	backend := &fakeBackend{status: old}
	m := home(t, backend)
	m, slow := update(t, m, tickMsg{})
	if m, _ = update(t, m, tickMsg{}); backend.count("status") != 1 {
		t.Fatal("a second poll started while the first was unanswered")
	}
	newer := statusFixture()
	newer.Recordings[0].Summary = "exported"
	backend.status = newer
	next, cmd := m.reload()
	m = settle(t, next.(Model), cmd)
	backend.status = old
	m = settle(t, m, slow)
	if m.status.Recordings[0].Summary != "exported" || m.busy != "" || m.statusInFlight {
		t.Fatalf("stale refresh applied: %q busy=%q", m.status.Recordings[0].Summary, m.busy)
	}
}

func TestStatusFailureClearsOnceStatusReadsAgain(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	backend.statusErr = errors.New("eegfeat preprocess status: exit 1\neegfeat preprocess: study.yaml: mapping values are not allowed here")
	m, cmd := update(t, m, tickMsg{})
	m = settle(t, m, cmd)
	if !strings.Contains(m.View(), "mapping values are not allowed") || len(m.status.Recordings) != 4 {
		t.Fatal("a failed refresh must say why and keep the last status")
	}
	backend.statusErr = nil
	m, cmd = update(t, m, tickMsg{})
	m = settle(t, m, cmd)
	if strings.Contains(m.View(), "mapping values") {
		t.Fatal("a recovered status still shows the old failure")
	}
}

func TestSpanRejectsNonFiniteNumbers(t *testing.T) {
	for _, input := range []string{"NaN 1", "0 NaN", "Inf 1", "0 +Inf", "-Inf 1"} {
		t.Run(input, func(t *testing.T) {
			g := newGate("sub-01", rawGate())
			before := len(g.spans)
			g.addSpan(input)
			if g.problem == "" || len(g.spans) != before {
				t.Fatalf("invalid span accepted: %q", input)
			}
		})
	}
}

func TestSaveFinishesBeforeLeavingReview(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: icaGate()}
	m := openGate(t, backend)
	m, save := update(t, m, special(tea.KeyEnter))
	if !strings.Contains(m.View(), "saving") {
		t.Fatal("pending save is not visible")
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	m, reload := step(t, m, save)
	m = settle(t, m, reload)
	if m.gate != nil || m.busy != "" || len(backend.decisions) != 1 {
		t.Fatal("save did not finish and return home exactly once")
	}
}

func TestSpanValidationStaysVisibleAtEndOfLongReview(t *testing.T) {
	gate := rawGate()
	for i := 0; i < 80; i++ {
		gate.Spans = append(gate.Spans, eegfeat.Span{Onset: float64(i), Duration: 1, Description: "BAD_manual"})
	}
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: gate})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	m, _ = update(t, m, special(tea.KeyEnd))
	m, _ = update(t, m, special(tea.KeyEnter))
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("NaN 1")})
	m, _ = update(t, m, special(tea.KeyEnter))
	view := m.View()
	assertFocusedRow(t, view, "add")
	if !strings.Contains(view, "finite numbers") || !m.gate.editing || m.gate.input.Value() != "NaN 1" {
		t.Fatalf("invalid span must remain editable with a visible error:\n%s", view)
	}
}

func TestHomeAndEndNavigateBothHomePanes(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, special(tea.KeyEnd))
	assertFocusedRow(t, m.View(), "sub-04")
	m, _ = update(t, m, special(tea.KeyHome))
	assertFocusedRow(t, m.View(), "sub-01")
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyEnd))
	assertFocusedRow(t, m.View(), "export")
	m, _ = update(t, m, special(tea.KeyHome))
	assertFocusedRow(t, m.View(), "load")
}
