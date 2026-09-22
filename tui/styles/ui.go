package styles

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

const (
	MinWidth  = 100
	MinHeight = 24

	CheckMark   = "✓"
	PendingMark = "○"
	ActiveMark  = "●"
	WarningMark = "⚠"
	CrossMark   = "✗"
	Ellipsis    = "…"
)

var (
	Title   = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	Heading = lipgloss.NewStyle().Foreground(Text).Bold(true)
	Strong  = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	Value   = lipgloss.NewStyle().Foreground(Text)
	Dim     = lipgloss.NewStyle().Foreground(TextDim)
	Faint   = lipgloss.NewStyle().Foreground(Muted)
	Warn    = lipgloss.NewStyle().Foreground(Warning)
	Fail    = lipgloss.NewStyle().Foreground(Error)
	rule    = lipgloss.NewStyle().Foreground(Secondary)
	dot     = lipgloss.NewStyle().Foreground(Border)

	panel        = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(Border).Padding(1, 2)
	panelFocused = panel.BorderForeground(BorderBright)

	keyBracket   = lipgloss.NewStyle().Foreground(Border)
	keyPrimary   = lipgloss.NewStyle().Foreground(Primary).Bold(true)
	keySecondary = lipgloss.NewStyle().Foreground(TextDim).Bold(true)
)

type Hint struct {
	Key     string
	Label   string
	Primary bool
}

func RenderHint(h Hint) string {
	key, label := keySecondary, Faint
	if h.Primary {
		key, label = keyPrimary, Value
	}
	return keyBracket.Render("[") + key.Render(h.Key) + keyBracket.Render("] ") + label.Render(h.Label)
}

// RenderFooterHints keeps primary shortcuts in fixed-width slots. The complete
// contextual key list belongs in the help view, not in the footer.
func RenderFooterHints(width int, hints []Hint) string {
	parts := make([]string, 0, len(hints))
	for _, h := range hints {
		if h.Primary {
			text := RenderHint(h)
			parts = append(parts, text+strings.Repeat(" ", max(14-lipgloss.Width(text), 0)))
		}
	}
	return Truncate(strings.TrimRight(strings.Join(parts, "  "), " "), width)
}

func RenderRule(width int) string {
	return rule.Render(strings.Repeat("─", max(width, 0)))
}

func RenderHeading(title string) string {
	return Heading.Render(strings.ToUpper(title))
}

func RenderLabelValue(label, value string) string {
	return Faint.Render(label) + dot.Render(" · ") + Heading.Render(value)
}

func RenderCheckbox(checked, focused bool) string {
	glyph, style := "□", Faint
	if checked {
		glyph, style = "▣", Value
	}
	if focused {
		style = Strong
	}
	return style.Render(glyph)
}

func RenderCursor(focused bool) string {
	if focused {
		return Strong.Render("▎")
	}
	return " "
}

// ClampBlock truncates every line to width and pads it back out, so a panel's
// content never wraps or shifts.
func ClampBlock(content string, width int) string {
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		if lipgloss.Width(line) > width {
			line = ansi.Truncate(line, width, Ellipsis)
		}
		lines[i] = line + strings.Repeat(" ", max(0, width-lipgloss.Width(line)))
	}
	return strings.Join(lines, "\n")
}

func Truncate(text string, width int) string {
	if lipgloss.Width(text) <= width {
		return text
	}
	return ansi.Truncate(text, width, Ellipsis)
}

// RenderPanel draws content inside a bordered panel of exactly width × height cells.
func RenderPanel(focused bool, width, height int, content string) string {
	style := panel
	if focused {
		style = panelFocused
	}
	inner := max(width-style.GetHorizontalFrameSize(), 1)
	rows := max(height-style.GetVerticalFrameSize(), 1)
	lines := strings.Split(content, "\n")
	if len(lines) > rows {
		lines = lines[:rows]
	}
	block := ClampBlock(strings.Join(lines, "\n"), inner)
	return style.Width(width - style.GetHorizontalBorderSize()).Height(height - style.GetVerticalBorderSize()).Render(block)
}

func IsTooSmall(width, height int) bool {
	return width < MinWidth || height < MinHeight
}

func RenderTooSmall(width, height int) string {
	return Warn.Bold(true).Render(WarningMark+" Terminal too small") + "\n" +
		Dim.Render(fmt.Sprintf("Resize to at least %dx%d", MinWidth, MinHeight)) + "\n" +
		Faint.Render(fmt.Sprintf("Current: %dx%d · Ctrl+C to quit", width, height))
}
