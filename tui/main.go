// eegtable-tui: an optional terminal front end for `eegtable preprocess`.
package main

import (
	"flag"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/JoshuaDuq/EEGTable/tui/app"
	"github.com/JoshuaDuq/EEGTable/tui/eegtable"
	"github.com/JoshuaDuq/EEGTable/tui/styles"
)

func main() {
	jobs := flag.Int("n-jobs", 1, "parallel jobs handed to `eegtable preprocess run`")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: eegtable-tui [--n-jobs N] preprocessing.yaml")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	config := flag.Arg(0)
	if _, err := os.Stat(config); err != nil {
		fail(err)
	}
	binary, err := eegtable.Locate()
	if err != nil {
		fail(err)
	}
	styles.ApplyNoColorProfile()
	client := eegtable.Client{Binary: binary, Config: config, Jobs: *jobs}
	if _, err := tea.NewProgram(app.New(client, config), tea.WithAltScreen()).Run(); err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "eegtable-tui:", err)
	os.Exit(2)
}
