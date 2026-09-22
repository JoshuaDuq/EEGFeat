package app

import (
	"strings"

	"github.com/JoshuaDuq/EEGFeatML/tui/styles"
)

func (m Model) viewHelp() string {
	hints := m.homeHints()
	if m.logOpen {
		hints = logHints()
	} else if m.gate != nil {
		hints = m.gate.hints()
	}
	lines := make([]string, 0, len(hints))
	for _, binding := range hints {
		lines = append(lines, styles.RenderHint(binding))
	}
	return m.header(styles.Title.Render("HELP"), styles.Dim.Render("Keyboard shortcuts")) + "\n" +
		styles.RenderPanel(true, m.width-2, m.height-4, strings.Join(lines, "\n")) + "\n" +
		m.footer([]styles.Hint{hint("esc", "Back")})
}
