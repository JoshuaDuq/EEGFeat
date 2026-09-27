(function () {
  document.querySelectorAll("dl.py.class > dd > p:first-child").forEach(function (paragraph) {
    var first = paragraph.firstChild;
    var text = first && first.nodeType === Node.TEXT_NODE ? first.textContent : "";
    if (text.trim().indexOf("Bases:") === 0) paragraph.classList.add("api-bases");
  });

  // Tag sideways scrollers for the edge fades in custom.css. A region that
  // scrolls also needs to be reachable by keyboard.
  var scrollers = document.querySelectorAll("article .table-wrapper, article div.highlight pre");

  function markOverflow(el) {
    var overflowing = el.scrollWidth > el.clientWidth + 1;
    el.classList.toggle("is-overflowing", overflowing);
    if (overflowing) {
      el.classList.toggle("at-start", el.scrollLeft <= 1);
      el.classList.toggle("at-end", el.scrollLeft + el.clientWidth >= el.scrollWidth - 1);
      if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    }
  }

  function markAll() {
    scrollers.forEach(markOverflow);
  }

  scrollers.forEach(function (el) {
    el.addEventListener("scroll", function () { markOverflow(el); }, { passive: true });
  });
  window.addEventListener("resize", markAll);
  markAll();
  // Web fonts change line widths once they arrive.
  if (document.fonts) document.fonts.ready.then(markAll);

  // "auto" renders the dark palette (custom.css), so Furo's three-state
  // auto -> light -> dark cycle had a click that changed nothing. Capture
  // the click before Furo's handler and flip between the two real themes.
  document.addEventListener(
    "click",
    function (event) {
      if (!event.target.closest(".theme-toggle")) return;
      event.stopPropagation();
      var next = document.body.dataset.theme === "light" ? "dark" : "light";
      document.body.dataset.theme = next;
      try {
        localStorage.setItem("theme", next);
      } catch (error) {}
    },
    true,
  );

  var search = document.querySelector(".sidebar-search");
  if (!search) return;

  function openSearch() {
    var drawer = document.querySelector(".sidebar-drawer");
    var toggle = document.getElementById("__navigation");
    if (drawer && toggle && getComputedStyle(drawer).position === "fixed") {
      toggle.checked = true;
    }
    search.focus();
  }

  document.addEventListener("keydown", function (event) {
    var key = event.key;
    // Furo's drawers are checkbox toggles with no keyboard way out.
    if (key === "Escape") {
      ["__navigation", "__toc"].forEach(function (id) {
        var toggle = document.getElementById(id);
        if (toggle && toggle.checked) {
          toggle.checked = false;
          if (document.activeElement) document.activeElement.blur();
        }
      });
      return;
    }
    var field = event.target.closest("input, textarea, select, [contenteditable='true']");
    if (key === "/" && !field && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      openSearch();
      return;
    }
    if ((event.metaKey || event.ctrlKey) && key.toLowerCase() === "k") {
      event.preventDefault();
      openSearch();
    }
  });
})();
