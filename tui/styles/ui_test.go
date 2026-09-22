package styles

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestClampBlockTruncatesAndPadsEveryLine(t *testing.T) {
	got := ClampBlock("short\na line that is far too long", 10)
	lines := strings.Split(got, "\n")
	if len(lines) != 2 || lines[0] != "short     " || lines[1] != "a line th…" {
		t.Fatalf("got %q", lines)
	}
}

func TestFooterKeepsPrimaryShortcutsInStableSlots(t *testing.T) {
	hints := []Hint{{"↵", "Save", true}, {"o", "Sort by score", false}, {"esc", "Back", true}}
	wide := RenderFooterHints(160, hints)
	narrow := RenderFooterHints(98, hints)
	if wide != narrow || strings.Contains(wide, "Sort") {
		t.Fatalf("footer changed its shortcut list with width: %q / %q", wide, narrow)
	}
	hints[0].Label = "Add span"
	changed := RenderFooterHints(98, hints)
	if strings.Index(changed, "[esc]") != strings.Index(wide, "[esc]") {
		t.Fatal("action label moved the following shortcut")
	}
	if lipgloss.Width(changed) > 98 {
		t.Fatal("footer exceeds available width")
	}
}

func TestTooSmallNamesTheMinimum(t *testing.T) {
	if !IsTooSmall(50, 40) || IsTooSmall(100, 30) {
		t.Fatal("width threshold")
	}
	if !strings.Contains(RenderTooSmall(50, 10), "100x24") {
		t.Fatal("message must name the minimum size")
	}
}
