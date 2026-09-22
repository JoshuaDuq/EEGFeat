package app

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/JoshuaDuq/EEGFeatML/tui/eegfeat"
	"github.com/JoshuaDuq/EEGFeatML/tui/styles"
)

// gateModel is one checklist: the gate's items in display order, then its
// spans, then (raw gate only) a row that adds a span.
type gateModel struct {
	recording string
	gate      eegfeat.Gate
	checked   []bool
	spans     []spanRow
	order     []int
	sorted    bool
	cursor    int
	input     textinput.Model
	editing   bool
	problem   string
}

type spanRow struct {
	eegfeat.Span
	kept bool
}

func newGate(recording string, gate eegfeat.Gate) gateModel {
	g := gateModel{recording: recording, gate: gate, checked: make([]bool, len(gate.Items)), order: make([]int, len(gate.Items))}
	for i := range gate.Items {
		g.order[i] = i
	}
	for _, span := range gate.Spans {
		g.spans = append(g.spans, spanRow{span, true})
	}
	g.input = textinput.New()
	g.input.Prompt = ""
	g.input.Placeholder = "onset duration [description]"
	g.restore()
	return g
}

func (g *gateModel) restore() {
	for i, item := range g.gate.Items {
		g.checked[i] = item.Suggested
	}
	for i, span := range g.gate.Spans {
		g.spans[i].kept = span.Suggested
	}
}

func (g *gateModel) clear() {
	for i := range g.checked {
		g.checked[i] = false
	}
}

// edited is whether leaving would lose anything: the gate opens on the suggestion.
func (g gateModel) edited() bool {
	if len(g.spans) != len(g.gate.Spans) {
		return true
	}
	for i, item := range g.gate.Items {
		if g.checked[i] != item.Suggested {
			return true
		}
	}
	for i, span := range g.gate.Spans {
		if g.spans[i].kept != span.Suggested {
			return true
		}
	}
	return false
}

func (g gateModel) hasSpans() bool { return g.gate.Field == "bads" }

func (g gateModel) rows() int {
	n := len(g.gate.Items) + len(g.spans)
	if g.hasSpans() {
		n++
	}
	return n
}

func (g gateModel) onAddRow() bool { return g.hasSpans() && g.cursor == g.rows()-1 }

func (g *gateModel) toggle() {
	switch {
	case g.cursor < len(g.gate.Items):
		index := g.order[g.cursor]
		g.checked[index] = !g.checked[index]
	case g.cursor < len(g.gate.Items)+len(g.spans):
		span := &g.spans[g.cursor-len(g.gate.Items)]
		span.kept = !span.kept
	}
}

func (g gateModel) scored() bool {
	for _, item := range g.gate.Items {
		if item.Score != nil {
			return true
		}
	}
	return false
}

func (g *gateModel) sortByScore() {
	selected := -1
	if g.cursor < len(g.order) {
		selected = g.order[g.cursor]
	}
	g.sorted = !g.sorted
	for i := range g.order {
		g.order[i] = i
	}
	if g.sorted {
		items := g.gate.Items
		sort.SliceStable(g.order, func(a, b int) bool {
			x, y := items[g.order[a]].Score, items[g.order[b]].Score
			switch {
			case x == nil:
				return false
			case y == nil:
				return true
			}
			return *x > *y
		})
	}
	if selected >= 0 {
		g.cursor = slices.Index(g.order, selected)
	}
}

// addSpan parses "onset duration [description]" against the recording length.
func (g *gateModel) addSpan(text string) {
	fields := strings.Fields(text)
	if len(fields) < 2 || len(fields) > 3 {
		g.problem = "expected: onset duration [description]"
		return
	}
	onset, err1 := strconv.ParseFloat(fields[0], 64)
	duration, err2 := strconv.ParseFloat(fields[1], 64)
	description := "BAD_manual"
	if len(fields) == 3 {
		description = fields[2]
	}
	switch {
	case err1 != nil || err2 != nil:
		g.problem = "expected: onset duration [description], in seconds"
	case math.IsNaN(onset) || math.IsInf(onset, 0) || math.IsNaN(duration) || math.IsInf(duration, 0):
		g.problem = "onset and duration must be finite numbers"
	case onset < 0 || duration <= 0:
		g.problem = "onset must be ≥ 0 and duration > 0"
	case onset+duration > g.gate.Duration:
		g.problem = fmt.Sprintf("span ends at %.2f s, past the recording (%.1f s)", onset+duration, g.gate.Duration)
	case !strings.HasPrefix(strings.ToUpper(description), "BAD"):
		g.problem = "description must start with BAD"
	default:
		g.problem = ""
		g.spans = append(g.spans, spanRow{eegfeat.Span{Onset: onset, Duration: duration, Description: description}, true})
		g.cursor = len(g.gate.Items) + len(g.spans) - 1
	}
}

func (g gateModel) decision() eegfeat.Decision {
	d := eegfeat.Decision{ParentID: g.gate.ParentID, FitID: g.gate.FitID, Field: g.gate.Field}
	for i, item := range g.gate.Items {
		if g.checked[i] {
			d.IDs = append(d.IDs, item.ID)
		}
	}
	for _, span := range g.spans {
		if span.kept {
			d.Spans = append(d.Spans, eegfeat.Span{Onset: span.Onset, Duration: span.Duration, Description: span.Description})
		}
	}
	return d
}

func (m Model) updateGate(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	g := m.gate
	if g.editing {
		switch msg.String() {
		case "enter":
			g.addSpan(g.input.Value())
			if g.problem == "" {
				g.input.Reset()
				g.editing = false
				g.input.Blur()
			}
		case "esc":
			g.editing, g.problem = false, ""
			g.input.Reset()
			g.input.Blur()
		default:
			var cmd tea.Cmd
			g.input, cmd = g.input.Update(msg)
			return m, cmd
		}
		return m, nil
	}
	switch msg.String() {
	case "q":
		return m.requestQuit()
	case "esc":
		if g.edited() {
			m.confirm = &confirm{"Discard the changes to " + g.gate.Stage + "?", "", "Discard", Model.closeGate}
			return m, nil
		}
		return m.closeGate()
	case "up", "down":
		step := -1
		if msg.String() == "down" {
			step = 1
		}
		g.cursor = clamp(g.cursor+step, g.rows())
	case "pgup":
		g.cursor = clamp(g.cursor-max(m.height-12, 1), g.rows())
	case "pgdown":
		g.cursor = clamp(g.cursor+max(m.height-12, 1), g.rows())
	case "home":
		g.cursor = 0
	case "end":
		g.cursor = clamp(g.rows()-1, g.rows())
	case " ":
		if g.onAddRow() {
			return m, g.edit()
		}
		g.toggle()
	case "s":
		g.restore()
	case "a":
		g.clear()
	case "o":
		if g.scored() {
			g.sortByScore()
		}
	case "v":
		return m, m.viewer(g.recording, g.gate.Parent)
	case "enter":
		if g.onAddRow() {
			return m, g.edit()
		}
		m.busy, m.failure = "saving", ""
		return m, m.save()
	}
	return m, nil
}

func (m Model) closeGate() (tea.Model, tea.Cmd) {
	m.gate = nil
	return m.reload()
}

func (g *gateModel) edit() tea.Cmd {
	g.editing = true
	return g.input.Focus()
}

func (m Model) viewGate() string {
	g := m.gate
	title := styles.Title.Render(g.recording) + styles.Faint.Render(" · ") + styles.Value.Render(g.gate.Stage)
	summary := g.summary()
	if m.busy != "" {
		summary = m.busy + "…"
	}
	header := m.header(title, styles.Dim.Render(summary))
	rows := m.height - 8
	message := g.problem
	if m.failure != "" {
		message = m.failure
	}
	if message != "" {
		message = ansi.Wrap(message, m.width-8, "")
		rows -= lipgloss.Height(message) + 1
	}
	content := g.body(m.width-8, max(rows, 3))
	if message != "" {
		content += "\n" + styles.Warn.Render(message)
	}
	body := styles.RenderPanel(true, m.width-2, m.height-4, content)
	return header + "\n" + body + "\n" + m.footer(g.hints())
}

func (g gateModel) section() string {
	switch {
	case g.gate.Field == "bads":
		return "channels"
	case g.gate.Field == "include":
		return "projectors"
	case g.gate.Field == "apply":
		return "regression"
	case g.gate.Stage == "review-epochs":
		return "epochs"
	}
	return "components"
}

func (g gateModel) verb() string {
	switch g.gate.Field {
	case "bads":
		return "marked bad"
	case "include", "apply":
		return "applied"
	}
	return "excluded"
}

func (g gateModel) summary() string {
	ticked := 0
	for _, checked := range g.checked {
		if checked {
			ticked++
		}
	}
	text := fmt.Sprintf("%d %s · %d %s", len(g.gate.Items), g.section(), ticked, g.verb())
	if g.hasSpans() {
		kept := 0
		for _, span := range g.spans {
			if span.kept {
				kept++
			}
		}
		text += " · " + plural(kept, "span") + " kept"
	}
	if g.gate.Method != "" {
		text = strings.ToUpper(g.gate.Method) + " · " + text
	}
	return text
}

func (g gateModel) body(width, rows int) string {
	labelWidth, scored := 0, g.scored()
	heading := styles.RenderHeading(g.section())
	if scored {
		heading += styles.Faint.Render("  " + g.scoreMeaning())
	}
	lines := []string{heading, ""}
	focus := g.cursor + 2
	for _, item := range g.gate.Items {
		labelWidth = max(labelWidth, lipgloss.Width(item.Label))
	}
	labelWidth = min(labelWidth, width/2)
	for row, index := range g.order {
		item := g.gate.Items[index]
		focused := row == g.cursor
		line := styles.RenderCursor(focused) + " " + styles.RenderCheckbox(g.checked[index], focused) + " " + styles.Value.Render(pad(styles.Truncate(item.Label, labelWidth), labelWidth))
		if scored {
			line += "  " + styles.Dim.Render(pad(formatScore(item.Score), 6))
		}
		lines = append(lines, styles.Truncate(line+"  "+styles.Faint.Render(strings.Join(item.Tags, " · ")), width))
	}
	if g.hasSpans() {
		if g.cursor >= len(g.gate.Items) {
			focus += 2
		}
		lines = append(lines, "", styles.RenderHeading("spans")+styles.Faint.Render("  seconds from the start of the recording"))
		for i, span := range g.spans {
			focused := g.cursor == len(g.gate.Items)+i
			lines = append(lines, styles.RenderCursor(focused)+" "+styles.RenderCheckbox(span.kept, focused)+" "+
				styles.Value.Render(fmt.Sprintf("%8.2f  +%.2f", span.Onset, span.Duration))+"  "+styles.Faint.Render(span.Description))
		}
		focused := g.onAddRow()
		add := styles.Faint.Render("add   ") + styles.Faint.Render(g.input.Placeholder)
		if g.editing {
			input := g.input
			input.Width = max(width-11, 1)
			input.SetCursor(input.Position())
			add = styles.Faint.Render("add   ") + input.View()
		}
		lines = append(lines, styles.RenderCursor(focused)+"   "+add)
	}
	return scrollLines(lines, focus, rows)
}

// scoreMeaning names the column: only ICA components and epochs carry scores.
func (g gateModel) scoreMeaning() string {
	if g.gate.Stage == "review-epochs" {
		return "score: worst-channel peak-to-peak, µV"
	}
	return "score: ICLabel confidence"
}

func formatScore(score *float64) string {
	switch {
	case score == nil:
		return ""
	case *score <= 1:
		return fmt.Sprintf("%.2f", *score)
	}
	return fmt.Sprintf("%.1f", *score)
}

func (g gateModel) hints() []styles.Hint {
	if g.editing {
		return []styles.Hint{hint("↵", "Add span"), hint("esc", "Cancel")}
	}
	action := hint("↵", "Save")
	toggle := hint("space", "Toggle")
	if g.onAddRow() {
		action = hint("↵", "Add span")
		toggle = hint("space", "Add")
	}
	hints := []styles.Hint{action, toggle, minor("s", "Restore suggested"), minor("a", "Clear "+g.section())}
	if g.scored() {
		label := "Sort by score"
		if g.sorted {
			label = "Restore order"
		}
		hints = append(hints, minor("o", label))
	}
	return append(hints, hint("v", "Viewer"), hint("esc", "Back"), minor("pgup/pgdn", "Page"), minor("home/end", "Jump"))
}
