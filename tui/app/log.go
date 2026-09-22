package app

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/JoshuaDuq/EEGFeatML/tui/styles"
)

// logCap bounds what a run can hold: MNE narrates enough per stage that a whole
// run runs to thousands of lines, and a line costs about a hundred bytes.
const logCap = 20000

// logView keeps what Python wrote during runs; scroll counts lines back from the newest.
type logView struct {
	lines  []string
	scroll int
}

// add keeps one entry per line: Python's messages carry newlines of their own,
// and the panel counts entries when it decides what fits.
func (l *logView) add(text string) {
	lines := strings.Split(text, "\n")
	if l.scroll > 0 {
		l.scroll += len(lines)
	}
	l.lines = append(l.lines, lines...)
	if len(l.lines) > logCap {
		l.lines = l.lines[len(l.lines)-logCap:]
	}
	l.scroll = min(l.scroll, len(l.lines))
}

func (l *logView) mark(text string) {
	l.add(styles.Faint.Render("── " + text + " ──"))
}

func (l *logView) page(delta, rows int) {
	l.move(delta*rows, rows)
}

func (l *logView) move(delta, rows int) {
	limit := max(len(l.lines)-rows, 0)
	l.scroll = max(0, min(min(l.scroll, limit)+delta, limit))
}

func (l logView) view(rows int) string {
	end := max(len(l.lines)-l.scroll, min(rows, len(l.lines)))
	start := max(end-rows, 0)
	return strings.Join(l.lines[start:end], "\n")
}

// split gives the stage list the rows it needs and the log whatever is left,
// unless that would be too little to read.
func (m Model) split(available int) (panes, log int) {
	if m.run == nil && len(m.log.lines) == 0 {
		return available, 0
	}
	needed := len(m.visibleStages()) + 11
	panes = min(available, max(needed, 16))
	if log = available - panes; log < 5 {
		return available, 0
	}
	return panes, log
}

func (m Model) logRows() int {
	_, height := m.split(m.height - 4)
	return max(height-5, 0)
}

func (m Model) updateLog(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	rows := max(m.height-8, 1)
	switch msg.String() {
	case "esc", "l":
		m.logOpen = false
	case "up":
		m.log.move(1, rows)
	case "down":
		m.log.move(-1, rows)
	case "pgup":
		m.log.page(1, rows)
	case "pgdown":
		m.log.page(-1, rows)
	case "home":
		m.log.scroll = max(len(m.log.lines)-rows, 0)
	case "end":
		m.log.scroll = 0
	case "q":
		return m.quit()
	}
	return m, nil
}

func logHints() []styles.Hint {
	return []styles.Hint{hint("↑↓", "Scroll"), hint("pgup/pgdn", "Page"),
		hint("end", "Latest"), hint("esc", "Back"), minor("home", "Oldest"), minor("q", "Quit")}
}

func (m Model) viewLog() string {
	state := "Latest"
	if m.log.scroll > 0 {
		state = "Scrollback"
	}
	content := m.log.view(m.height - 8)
	if len(m.log.lines) == 0 {
		content = styles.Faint.Render("No run output yet.")
	}
	return m.header(styles.Title.Render("LOG"), styles.Dim.Render(state)) + "\n" +
		styles.RenderPanel(true, m.width-2, m.height-4, content) + "\n" + m.footer(logHints())
}
