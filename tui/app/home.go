package app

import (
	"fmt"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/JoshuaDuq/EEGFeatML/tui/eegfeat"
	"github.com/JoshuaDuq/EEGFeatML/tui/styles"
)

func (m Model) updateHome(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := msg.String()
	recording, ok := m.selected()
	switch key {
	case "l":
		m.logOpen = true
	case "q":
		return m.requestQuit()
	case "esc":
		if m.run != nil {
			m.confirm = &confirm{"Stop the run?", runDetail, "Stop run", Model.stop}
		} else {
			m.stagePane = false
		}
	case "tab":
		m.stagePane = !m.stagePane
	case "up", "down":
		m.move(key == "down")
	case "home", "end":
		if m.stagePane {
			m.stageCursor = 0
			if key == "end" {
				m.stageCursor = clamp(len(m.visibleStages())-1, len(m.visibleStages()))
			}
		} else {
			m.cursor, m.stageCursor = 0, 0
			if key == "end" {
				m.cursor = clamp(len(m.status.Recordings)-1, len(m.status.Recordings))
			}
		}
	case "pgup":
		m.log.page(1, m.logRows())
	case "pgdown":
		m.log.page(-1, m.logRows())
	case "enter":
		if !ok || m.run != nil {
			return m, nil
		}
		if m.stagePane {
			if stage := m.selectedStage(); stage.State == "needs-review" {
				return m.begin("opening", m.openGate(recording.Label, stage.Stage))
			}
			return m, nil
		}
		return m.next(recording)
	case "r":
		if ok && m.run == nil {
			return m.begin("starting", m.startRun(recording.Label))
		}
	case "a":
		if m.run == nil {
			return m.begin("starting", m.startRun(""))
		}
	case "x":
		if ok && m.stagePane && m.run == nil && m.selectedStage().Stage != "" {
			m.askReset(recording.Label, m.selectedStage().Stage)
		}
	case "v":
		if ok && m.stagePane && m.run == nil && m.selectedStage().State == "completed" {
			return m, m.viewer(recording.Label, m.selectedStage().Stage)
		}
	}
	return m, nil
}

func (m Model) begin(busy string, cmd tea.Cmd) (tea.Model, tea.Cmd) {
	m.busy, m.failure, m.notice = busy, "", ""
	return m, cmd
}

// next performs Python's own recommendation for the selected recording.
func (m Model) next(recording eegfeat.Recording) (tea.Model, tea.Cmd) {
	action := recording.Next
	switch {
	case action == nil:
		return m, nil
	case action.Kind == "review":
		return m.begin("opening", m.openGate(recording.Label, action.Stage))
	case action.Kind == "reset":
		m.askReset(recording.Label, action.Stage)
		return m, nil
	case m.run == nil:
		return m.begin("starting", m.startRun(recording.Label))
	}
	return m, nil
}

func (m *Model) askReset(recording, stage string) {
	m.confirm = &confirm{
		fmt.Sprintf("Reset %s from %s?", recording, stage),
		"Retires that stage and every later one; payloads stay on disk.",
		"Reset",
		func(m Model) (tea.Model, tea.Cmd) { return m.begin("resetting", m.reset(recording, stage)) },
	}
}

func (m Model) stop() (tea.Model, tea.Cmd) {
	if m.run != nil {
		m.run.runner.Stop()
		m.notice = "Stopping…"
	}
	return m, nil
}

func (m *Model) move(down bool) {
	step := -1
	if down {
		step = 1
	}
	if m.stagePane {
		m.stageCursor = clamp(m.stageCursor+step, len(m.visibleStages()))
	} else {
		m.cursor = clamp(m.cursor+step, len(m.status.Recordings))
		m.stageCursor = 0
	}
}

func clamp(index, count int) int {
	return max(0, min(index, count-1))
}

func (m Model) selected() (eegfeat.Recording, bool) {
	if m.cursor < len(m.status.Recordings) {
		return m.status.Recordings[m.cursor], true
	}
	return eegfeat.Recording{}, false
}

func (m Model) visibleStages() []eegfeat.Stage {
	recording, ok := m.selected()
	if !ok {
		return nil
	}
	stages := make([]eegfeat.Stage, 0, len(recording.Stages))
	for _, stage := range recording.Stages {
		if stage.State != "disabled" {
			stages = append(stages, stage)
		}
	}
	return stages
}

func (m Model) selectedStage() eegfeat.Stage {
	stages := m.visibleStages()
	if len(stages) == 0 {
		return eegfeat.Stage{}
	}
	return stages[clamp(m.stageCursor, len(stages))]
}

func (m Model) homeHeader() string {
	count := m.headerRight()
	return m.header(m.recipeTitle(m.width-lipgloss.Width(count)-4), count)
}

func (m Model) viewConfirm() string {
	content := styles.RenderHeading("Confirm") + "\n\n" + styles.Warn.Render(m.confirm.question)
	if m.confirm.detail != "" {
		content += "\n" + styles.Dim.Render(m.confirm.detail)
	}
	content = ansi.Wrap(content, m.width-8, "")
	return m.homeHeader() + "\n" + styles.RenderPanel(true, m.width-2, m.height-4, content) + "\n" +
		m.footer([]styles.Hint{hint("y", m.confirm.verb), hint("n", "Cancel")})
}

func (m Model) viewHome() string {
	width := m.width - 2
	header := m.homeHeader()
	paneHeight, logHeight := m.split(m.height - 4)
	left := m.leftWidth(width)
	right := width - left - 1
	body := lipgloss.JoinHorizontal(lipgloss.Top,
		styles.RenderPanel(!m.stagePane, left, paneHeight, m.recordingsPane(left-6)),
		" ",
		styles.RenderPanel(m.stagePane, right, paneHeight, m.stagesPane(right-6)),
	)
	if logHeight > 0 {
		content := styles.RenderHeading("log") + "\n" + m.log.view(logHeight-5)
		body += "\n" + styles.RenderPanel(false, width, logHeight, content)
	}
	return header + "\n" + body + "\n" + m.footer(m.homeFooterHints())
}

// recipeTitle keeps the recipe's file name whole and shortens its directory from the left.
func (m Model) recipeTitle(available int) string {
	dir, base := filepath.Split(m.config)
	brand := styles.Title.Render("eegfeat") + styles.Faint.Render(" · ")
	room := available - lipgloss.Width(brand) - lipgloss.Width(base)
	if runes := []rune(dir); room < len(runes) {
		dir = ""
		if room > 4 {
			dir = styles.Ellipsis + string(runes[len(runes)-(room-1):])
		}
	}
	return brand + styles.Faint.Render(dir) + styles.Value.Render(base)
}

// leftWidth fits the longest recording and its summary when the terminal allows.
func (m Model) leftWidth(width int) int {
	needed := 34
	for _, recording := range m.status.Recordings {
		needed = max(needed, lipgloss.Width(recording.Label)+lipgloss.Width(recording.Summary)+12)
	}
	return min(needed, width*55/100)
}

func (m Model) header(left, right string) string {
	width := m.width - 2
	left = styles.Truncate(left, max(width-lipgloss.Width(right)-2, 1))
	gap := max(width-lipgloss.Width(left)-lipgloss.Width(right), 1)
	return " " + left + strings.Repeat(" ", gap) + right + "\n " + styles.RenderRule(width)
}

func (m Model) headerRight() string {
	if m.busy != "" {
		return styles.Faint.Render(m.busy + "…")
	}
	awaiting := 0
	for _, recording := range m.status.Recordings {
		if recording.Next != nil && recording.Next.Kind == "review" {
			awaiting++
		}
	}
	text := plural(len(m.status.Recordings), "recording")
	if awaiting > 0 {
		text += fmt.Sprintf(" · %d awaiting review", awaiting)
	}
	return styles.Dim.Render(text)
}

func (m Model) footer(hints []styles.Hint) string {
	if !m.helpOpen && m.confirm == nil && (m.gate == nil || !m.gate.editing) {
		hints = append([]styles.Hint{hint("?", "Help")}, hints...)
	}
	return " " + styles.Truncate(styles.RenderFooterHints(m.width-2, hints), m.width-2)
}

func (m Model) homeFooterHints() []styles.Hint {
	action := hint("↵", "Next")
	if m.run != nil {
		action = hint("esc", "Stop run")
	}
	return []styles.Hint{hint("↑↓", "Move"), hint("tab", "Pane"), action,
		hint("l", "Log"), hint("q", "Quit")}
}

func (m Model) homeHints() []styles.Hint {
	hints := []styles.Hint{hint("↑↓", "Navigate")}
	if m.run != nil {
		return append(hints, hint("esc", "Stop run"), hint("l", "Log"), minor("tab", "Switch pane"), minor("q", "Quit"))
	}
	if m.stagePane {
		stage := m.selectedStage()
		if stage.State == "needs-review" {
			hints = append(hints, hint("↵", "Review"))
		}
		if stage.State == "completed" {
			hints = append(hints, hint("v", "Viewer"))
		}
		if stage.Stage != "" {
			hints = append(hints, minor("x", "Reset from stage"))
		}
		hints = append(hints, hint("tab", "Recordings"), minor("esc", "Recordings"))
	} else {
		if recording, ok := m.selected(); ok {
			if recording.Next != nil {
				hints = append(hints, hint("↵", nextLabel(*recording.Next)))
			}
			hints = append(hints, minor("r", "Run"))
		}
		hints = append(hints, hint("tab", "Stages"), minor("a", "Run all"))
	}
	hints = append(hints, hint("l", "Log"), minor("home/end", "Jump"), minor("q", "Quit"))
	if m.logRows() > 0 && len(m.log.lines) > 0 {
		hints = append(hints, minor("pgup/pgdn", "Scroll log"))
	}
	return hints
}

func nextLabel(next eegfeat.Next) string {
	switch next.Kind {
	case "review":
		return "Review " + next.Target
	case "reset":
		return "Reset from " + next.Stage
	}
	return "Run"
}

func (m Model) recordingsPane(width int) string {
	lines := []string{styles.RenderHeading("recordings") + styles.Faint.Render(fmt.Sprintf("  %d", len(m.status.Recordings))), ""}
	labelWidth := 0
	for _, recording := range m.status.Recordings {
		labelWidth = max(labelWidth, lipgloss.Width(recording.Label))
	}
	labelWidth = min(labelWidth, width/2)
	for i, recording := range m.status.Recordings {
		focused := i == m.cursor
		summary := recording.Summary
		if m.run.isRunning(recording.Label) {
			summary = "running"
		}
		label := styles.Value.Render(pad(styles.Truncate(recording.Label, labelWidth), labelWidth))
		if focused {
			label = styles.Strong.Render(pad(styles.Truncate(recording.Label, labelWidth), labelWidth))
		}
		lines = append(lines, styles.RenderCursor(focused && !m.stagePane)+" "+label+"  "+styles.Dim.Render(summary))
	}
	if len(m.status.Recordings) == 0 && m.busy == "" {
		lines = append(lines, styles.Faint.Render("no recordings in this recipe"))
	}
	panes, _ := m.split(m.height - 4)
	return scrollLines(lines, m.cursor+2, panes-4)
}

func (m Model) stagesPane(width int) string {
	recording, ok := m.selected()
	if !ok {
		return ansi.Wrap(m.messages(), width, "")
	}
	lines := []string{styles.Heading.Render(recording.Label), ""}
	stages := m.visibleStages()
	nameWidth := 0
	for _, stage := range stages {
		nameWidth = max(nameWidth, len(stage.Stage))
	}
	running := m.run.isRunning(recording.Label)
	start, end := window(len(stages), m.stageCursor, m.stageRows())
	if start > 0 {
		lines = append(lines, styles.Faint.Render(fmt.Sprintf("    ↑ %d more", start)))
	}
	for i, stage := range stages {
		state := stage.State
		if m.run.done(recording.Label, stage.Stage) {
			state = "completed"
		}
		if running && state != "completed" {
			state, running = "running", false
		}
		if i < start || i >= end {
			continue
		}
		focused := m.stagePane && i == m.stageCursor
		lines = append(lines, styles.RenderCursor(focused)+" "+stateGlyph(state)+" "+styles.Value.Render(pad(stage.Stage, nameWidth))+"  "+stateText(state))
	}
	if end < len(stages) {
		lines = append(lines, styles.Faint.Render(fmt.Sprintf("    ↓ %d more", len(stages)-end)))
	}
	if recording.Next != nil && m.run == nil && !m.stagePane {
		lines = append(lines, "", styles.RenderLabelValue("NEXT", strings.ToLower(nextLabel(*recording.Next)))+"  "+styles.RenderHint(hint("↵", "")))
	}
	if extra := m.messages(); extra != "" {
		lines = append(lines, "", ansi.Wrap(extra, width, ""))
	}
	return strings.Join(lines, "\n")
}

// messages is the one place a failure or a notice is shown.
func (m Model) messages() string {
	switch {
	case m.failure != "":
		return styles.Fail.Render(m.failure)
	case m.statusErr != "":
		return styles.Fail.Render(m.statusErr)
	case m.notice != "":
		return styles.Dim.Render(m.notice)
	}
	return ""
}

// stageRows is how many stage lines fit beside the title, NEXT line and messages.
func (m Model) stageRows() int {
	panes, _ := m.split(m.height - 4)
	return max(panes-11, 3)
}

// window picks the slice of count rows that keeps cursor visible within rows,
// leaving a line for each "more" marker it needs.
func window(count, cursor, rows int) (int, int) {
	if count <= rows {
		return 0, count
	}
	rows = max(rows-2, 1)
	start := max(0, min(cursor-rows/2, count-rows))
	return start, min(start+rows, count)
}

func scrollLines(lines []string, cursor, rows int) string {
	start, end := window(len(lines), cursor, rows)
	visible := make([]string, 0, end-start+2)
	if start > 0 {
		visible = append(visible, styles.Faint.Render(fmt.Sprintf("    ↑ %d more", start)))
	}
	visible = append(visible, lines[start:end]...)
	if end < len(lines) {
		visible = append(visible, styles.Faint.Render(fmt.Sprintf("    ↓ %d more", len(lines)-end)))
	}
	return strings.Join(visible, "\n")
}

func stateGlyph(state string) string {
	switch state {
	case "completed":
		return styles.Value.Render(styles.CheckMark)
	case "needs-review", "running":
		return styles.Strong.Render(styles.ActiveMark)
	case "stale":
		return styles.Warn.Render(styles.WarningMark)
	}
	return styles.Faint.Render(styles.PendingMark)
}

func stateText(state string) string {
	text := strings.ReplaceAll(state, "-", " ")
	switch state {
	case "needs-review", "running":
		return styles.Strong.Render(text)
	case "stale":
		return styles.Warn.Render(text)
	case "completed":
		return styles.Dim.Render(text)
	}
	return styles.Faint.Render(text)
}

func plural(n int, noun string) string {
	if n == 1 {
		return "1 " + noun
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

func pad(text string, width int) string {
	return text + strings.Repeat(" ", max(0, width-lipgloss.Width(text)))
}
