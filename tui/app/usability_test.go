package app

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/JoshuaDuq/EEGFeatML/tui/eegfeat"
	"github.com/JoshuaDuq/EEGFeatML/tui/styles"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func TestLogCanBeOpenedAtMinimumSize(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 24)
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	m = feed(t, m, backend, eegfeat.Event{Event: "stderr", Message: "Filtering raw data"})
	m, _ = update(t, m, key('l'))
	view := m.View()
	if !strings.Contains(view, "Filtering raw data") || !strings.Contains(view, "Latest") {
		t.Fatalf("log must be accessible at minimum size:\n%s", view)
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	if backend.runner.stopped || !strings.Contains(m.View(), "RECORDINGS") {
		t.Fatal("closing the log must return home without stopping the run")
	}
}

func TestHelpRevealsHiddenActionsWithoutChangingSelection(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	m, _ = update(t, m, key('?'))
	view := m.View()
	for _, want := range []string{"HELP", "Run all", "Stages", "Quit"} {
		if !strings.Contains(view, want) {
			t.Fatalf("help lacks %q:\n%s", want, view)
		}
	}
	if lipgloss.Width(view) > 100 || lipgloss.Height(view) > 24 {
		t.Fatal("help overflows terminal")
	}
	m, cmd := update(t, m, key('r'))
	if cmd != nil {
		t.Fatal("help must not dispatch hidden actions")
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	if m.cursor != 0 || !strings.Contains(m.View(), "RECORDINGS") {
		t.Fatal("help changed selection")
	}
}

func TestStageHintsMatchAvailableActions(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, special(tea.KeyDown))
	hints := m.homeHints()
	found := false
	for _, hint := range hints {
		if hint.Key == "v" {
			t.Fatal("review gate advertises a nonexistent checkpoint viewer")
		}
		found = found || hint.Key == "↵"
	}
	if !found {
		t.Fatal("review gate must advertise Enter")
	}
}

func TestRunFailureSurvivesStatusRefresh(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	m, cmd := update(t, m, runDoneMsg{eegfeat.Result{Code: 1, Stderr: "ICA did not converge"}})
	m = settle(t, m, cmd)
	m, cmd = update(t, m, tickMsg{})
	m = settle(t, m, cmd)
	if !strings.Contains(m.View(), "ICA did not converge") {
		t.Fatal("refresh erased run failure")
	}
}

func TestScrolledLogStaysAnchoredWhenOutputArrives(t *testing.T) {
	log := logView{}
	log.add("one\ntwo\nthree\nfour")
	log.page(1, 2)
	before := log.view(2)
	log.add("five\nsix")
	if got := log.view(2); got != before {
		t.Fatalf("reading position moved: %q -> %q", before, got)
	}
}

func TestLogOldestPageRespondsImmediatelyToDown(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	for i := 0; i < 40; i++ {
		m.log.add(fmt.Sprintf("line %02d", i))
	}
	m, _ = update(t, m, key('l'))
	for i := 0; i < 60; i++ {
		m, _ = update(t, m, special(tea.KeyUp))
	}
	if !strings.Contains(m.View(), "line 00") {
		t.Fatal("oldest page is inaccessible")
	}
	m, _ = update(t, m, special(tea.KeyDown))
	if strings.Contains(m.View(), "line 00") || !strings.Contains(m.View(), "line 16") {
		t.Fatal("scrolling past the top created dead keypresses")
	}
	// Growing the viewport must also clamp its reading position before scrolling.
	m, _ = update(t, m, special(tea.KeyHome))
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 30})
	m, _ = update(t, m, special(tea.KeyDown))
	if strings.Contains(m.View(), "line 00") {
		t.Fatal("resize left the log overscrolled")
	}
}

func TestAddSpanRowHintsDescribeEnter(t *testing.T) {
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: rawGate()})
	m, _ = update(t, m, special(tea.KeyEnd))
	for _, binding := range m.gate.hints() {
		if binding.Key == "↵" && binding.Label != "Add span" {
			t.Fatal("Enter is incorrectly labeled Save")
		}
		if binding.Key == "space" && binding.Label == "Toggle" {
			t.Fatal("add row has no checkbox to toggle")
		}
	}
	m, _ = update(t, m, special(tea.KeyEnter))
	if !m.gate.editing {
		t.Fatal("Enter did not open the span input")
	}
}

func TestRefreshPreservesSelectedStageIdentity(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyDown))
	selected := m.selectedStage().Stage
	status := statusFixture()
	status.Recordings[0].Stages = append([]eegfeat.Stage{{Stage: "new-stage", State: "pending"}}, status.Recordings[0].Stages...)
	m, _ = update(t, m, statusMsg{seq: m.statusSeq, status: status})
	if m.selectedStage().Stage != selected {
		t.Fatal("refresh moved focus to a different stage")
	}
	status.Recordings = status.Recordings[1:]
	m, _ = update(t, m, statusMsg{seq: m.statusSeq, status: status})
	if m.stageCursor != 0 {
		t.Fatal("removed recording left an unrelated stage selected")
	}
}

func TestScrollingDoesNotMoveRunningStage(t *testing.T) {
	status := statusFixture()
	for i := 0; i < 20; i++ {
		status.Recordings[0].Stages = append(status.Recordings[0].Stages, eegfeat.Stage{Stage: fmt.Sprintf("later-%02d", i), State: "pending"})
	}
	backend := &fakeBackend{status: status}
	m := startRun(t, backend, 24)
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyEnd))
	for _, line := range strings.Split(m.View(), "\n") {
		if strings.Contains(line, "later-") && strings.Contains(line, "running") {
			t.Fatal("scrolling falsely marked a later stage as running")
		}
	}
}

func TestRunAllOnlyMarksTheActiveRecording(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, cmd := update(t, m, key('a'))
	m = settle(t, m, cmd)
	m = feed(t, m, backend, eegfeat.Event{Event: "subject_start", Subject: "sub-01"})
	for _, line := range strings.Split(m.View(), "\n") {
		if strings.Contains(line, "sub-02") && strings.Contains(line, "running") {
			t.Fatal("an inactive recording is marked running")
		}
	}
	m = feed(t, m, backend, eegfeat.Event{Event: "subject_done", Subject: "sub-01"})
	if strings.Contains(m.View(), "running") {
		t.Fatal("finished recording is still running")
	}
	m = feed(t, m, backend, eegfeat.Event{Event: "subject_start", Subject: "sub-04"})
	found := false
	for _, line := range strings.Split(m.View(), "\n") {
		found = found || strings.Contains(line, "sub-04") && strings.Contains(line, "running")
	}
	if !found {
		t.Fatal("new active recording was not marked running")
	}
}

func TestLongReviewLabelsKeepEvidenceVisible(t *testing.T) {
	gate := icaGate()
	gate.Items[0].Label = strings.Repeat("component-", 20)
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: gate})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	view := m.View()
	if !strings.Contains(view, "0.98") || !strings.Contains(view, "eye blink") {
		t.Fatal("long label hid the score or reason")
	}
}

func TestLongRecordingLabelsKeepStatusVisible(t *testing.T) {
	status := statusFixture()
	status.Recordings[0].Label = strings.Repeat("recording-", 20)
	m := home(t, &fakeBackend{status: status})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	if !strings.Contains(m.View(), "awaiting review-raw") {
		t.Fatal("long label hid recording status")
	}
}

func TestLongSpanInputKeepsCursorTextVisibleAcrossResize(t *testing.T) {
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: rawGate()})
	m, _ = update(t, m, special(tea.KeyEnd))
	m, _ = update(t, m, special(tea.KeyEnter))
	value := "1 2 BAD_" + strings.Repeat("annotation_", 20) + "TAIL"
	m, _ = update(t, m, tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(value)})
	for _, width := range []int{140, 100, 120} {
		m, _ = update(t, m, tea.WindowSizeMsg{Width: width, Height: 24})
		view := m.View()
		if !strings.Contains(view, "TAIL") {
			t.Fatalf("input cursor is hidden at width %d", width)
		}
		if lipgloss.Width(view) > width {
			t.Fatal("input exceeds terminal width")
		}
	}
	if m.gate.input.Value() != value {
		t.Fatal("display truncation changed the annotation")
	}
}

func TestRestoringSuggestionsPreservesManualSpanChoices(t *testing.T) {
	g := newGate("sub-01", rawGate())
	g.addSpan("10 1 BAD_manual")
	g.addSpan("12 1 BAD_manual")
	g.spans[0].kept = false
	g.spans[2].kept = false
	g.restore()
	if !g.spans[0].kept {
		t.Fatal("detector suggestion was not restored")
	}
	if !g.spans[1].kept || g.spans[2].kept {
		t.Fatal("restoring suggestions changed manual span choices")
	}
	if len(g.decision().Spans) != 2 {
		t.Fatal("manual annotation was lost from the decision")
	}
}

func TestSpanInputReceivesAsynchronousMessages(t *testing.T) {
	m := openGate(t, &fakeBackend{status: statusFixture(), gate: rawGate()})
	m, _ = update(t, m, special(tea.KeyEnd))
	m, _ = update(t, m, special(tea.KeyEnter))
	m.gate.input.Cursor.BlinkSpeed = time.Millisecond
	before := m.gate.input.Cursor.Blink
	blink := m.gate.input.Cursor.BlinkCmd()
	m, _ = update(t, m, blink())
	if m.gate.input.Cursor.Blink == before {
		t.Fatal("input component did not receive its asynchronous update")
	}
}

func TestSmallTerminalBlocksHiddenActions(t *testing.T) {
	for _, review := range []bool{false, true} {
		t.Run(fmt.Sprintf("review=%t", review), func(t *testing.T) {
			backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
			m := home(t, backend)
			if review {
				m = openGate(t, backend)
			}
			m, _ = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 20})
			for _, press := range []tea.KeyMsg{key('r'), key('a'), key(' '), special(tea.KeyEnter), special(tea.KeyDown), special(tea.KeyEsc)} {
				var cmd tea.Cmd
				m, cmd = update(t, m, press)
				if cmd != nil {
					t.Fatalf("hidden action dispatched for %s", press.String())
				}
			}
			if m.cursor != 0 {
				t.Fatal("hidden navigation changed recording")
			}
			if review && (m.gate == nil || m.gate.cursor != 0 || !m.gate.checked[0]) {
				t.Fatal("hidden keys changed the review")
			}
			m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
			_, cmd := update(t, m, special(tea.KeyEnter))
			if cmd == nil {
				t.Fatal("visible actions did not resume after resizing")
			}
		})
	}
}

func TestSmallTerminalKeepsSpanDraftAndRunAlive(t *testing.T) {
	backend := &fakeBackend{status: statusFixture(), gate: rawGate()}
	m := openGate(t, backend)
	m, _ = update(t, m, special(tea.KeyEnd))
	m, _ = update(t, m, special(tea.KeyEnter))
	m.gate.input.SetValue("1 2 BAD_manual")
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 20})
	for _, press := range []tea.KeyMsg{key('q'), special(tea.KeyEnter), special(tea.KeyEsc)} {
		var cmd tea.Cmd
		m, cmd = update(t, m, press)
		if cmd != nil {
			t.Fatal("hidden input dispatched an action")
		}
	}
	if !m.gate.editing || m.gate.input.Value() != "1 2 BAD_manual" {
		t.Fatal("resize lost the draft")
	}
	m = startRun(t, backend, 40)
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 80, Height: 20})
	m, _ = update(t, m, special(tea.KeyEsc))
	if backend.runner.stopped {
		t.Fatal("hidden Escape stopped the run")
	}
	m, cmd := update(t, m, special(tea.KeyCtrlC))
	if cmd == nil || !backend.runner.stopped {
		t.Fatal("Ctrl+C must still stop and quit")
	}
}

func TestIdleHomeUsesSpaceInsteadOfAnEmptyLogPanel(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	if strings.Contains(m.View(), "LOG") {
		t.Fatal("idle home shows an empty log panel")
	}
	panes, log := m.split(m.height - 4)
	if log != 0 || panes != m.height-4 {
		t.Fatal("empty log still takes space from the workflow")
	}
	m, _ = update(t, m, key('l'))
	if !strings.Contains(m.View(), "No run output yet.") {
		t.Fatal("explicit empty log has no explanation")
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	m, cmd := update(t, m, key('r'))
	m = settle(t, m, cmd)
	if !strings.Contains(m.View(), "LOG") {
		t.Fatal("starting a run did not reveal its log")
	}
	m, cmd = update(t, m, runDoneMsg{eegfeat.Result{Code: 3}})
	m = settle(t, m, cmd)
	if !strings.Contains(m.View(), "LOG") {
		t.Fatal("finished output should remain accessible on home")
	}
}

func TestResetConfirmationKeepsScopeVisibleForLongRecordingNames(t *testing.T) {
	status := statusFixture()
	status.Recordings[0].Label = "sub-" + strings.Repeat("long-recording-", 8) + "unique-end"
	backend := &fakeBackend{status: status}
	m := home(t, backend)
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	m, _ = update(t, m, special(tea.KeyTab))
	m, _ = update(t, m, special(tea.KeyDown))
	m, _ = update(t, m, key('x'))
	view := m.View()
	for _, want := range []string{"unique-end", "detect-bads?", "every later one", "payloads stay on disk", "[y] Reset", "[n] Cancel"} {
		if !strings.Contains(view, want) {
			t.Fatalf("reset confirmation hides %q:\n%s", want, view)
		}
	}
	if lipgloss.Width(view) > 100 || lipgloss.Height(view) > 24 {
		t.Fatal("confirmation exceeds the terminal")
	}
	m, cmd := update(t, m, special(tea.KeyEnter))
	if cmd != nil || m.confirm == nil {
		t.Fatal("Enter must not confirm a reset")
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	if m.confirm != nil || !m.stagePane || m.selectedStage().Stage != "detect-bads" {
		t.Fatal("cancel did not preserve the stage selection")
	}
}

func TestBackgroundRefreshAllowsBrowsingWithoutDispatchingActions(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, refresh := update(t, m, tickMsg{})
	m, _ = update(t, m, special(tea.KeyDown))
	if m.selectedLabel() != "sub-02" {
		t.Fatal("background refresh blocked navigation")
	}
	m, _ = update(t, m, special(tea.KeyTab))
	if !m.stagePane {
		t.Fatal("background refresh blocked pane switching")
	}
	m, _ = update(t, m, key('l'))
	if !m.logOpen {
		t.Fatal("background refresh blocked log access")
	}
	m, _ = update(t, m, special(tea.KeyEsc))
	m = settle(t, m, refresh)
	if m.selectedLabel() != "sub-02" || !m.stagePane {
		t.Fatal("refresh discarded browsing position")
	}
}

func TestActionDuringBackgroundRefreshIsNotDropped(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := home(t, backend)
	m, refresh := update(t, m, tickMsg{})
	m, start := update(t, m, key('r'))
	if start == nil || m.busy != "starting" {
		t.Fatal("a refresh in flight swallowed the key")
	}
	m = settle(t, m, refresh)
	if m.busy != "starting" {
		t.Fatal("the refresh answer cleared another action's busy state")
	}
	if m = settle(t, m, start); m.run == nil {
		t.Fatal("run did not start")
	}
}

func TestViewerCannotSuspendAnActiveRun(t *testing.T) {
	backend := &fakeBackend{status: statusFixture()}
	m := startRun(t, backend, 40)
	m, _ = update(t, m, special(tea.KeyTab))
	_, cmd := update(t, m, key('v'))
	if cmd != nil || backend.called("viewer sub-01 load") {
		t.Fatal("viewer launched during an active run")
	}
}

func TestHomeFooterStaysStableAcrossSelectionAndRefresh(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m, _ = update(t, m, tea.WindowSizeMsg{Width: 100, Height: 24})
	footer := func(m Model) string {
		lines := strings.Split(m.View(), "\n")
		return lines[len(lines)-1]
	}
	initial := footer(m)
	for _, press := range []tea.KeyMsg{special(tea.KeyDown), special(tea.KeyDown), special(tea.KeyTab), special(tea.KeyEnd)} {
		m, _ = update(t, m, press)
		if footer(m) != initial {
			t.Fatal("selection rearranged the shortcut bar")
		}
	}
	m, _ = update(t, m, tickMsg{})
	if footer(m) != initial {
		t.Fatal("background refresh rearranged the shortcut bar")
	}
	for _, label := range []string{"Help", "Move", "Pane", "Next", "Log", "Quit"} {
		if !strings.Contains(initial, label) {
			t.Fatalf("footer lacks %s", label)
		}
	}
}

func TestLogPanelDoesNotMoveFooterVertically(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	before := lipgloss.Height(m.View())
	m.log.add("Run output")
	if after := lipgloss.Height(m.View()); after != before {
		t.Fatalf("log panel moved footer from row %d to %d", before, after)
	}
}

func TestCompactFootersFitWithoutWrapping(t *testing.T) {
	m := home(t, &fakeBackend{status: statusFixture()})
	m.width = 100
	raw := newGate("sub-01", rawGate())
	ica := newGate("sub-01", icaGate())
	for _, bindings := range [][]styles.Hint{m.homeFooterHints(), raw.hints(), ica.hints(), logHints()} {
		footer := m.footer(bindings)
		if strings.Contains(footer, "\n") || lipgloss.Width(footer) > 99 || strings.Contains(footer, "…") {
			t.Fatalf("footer must fit fully on one line: %q", footer)
		}
	}
}
