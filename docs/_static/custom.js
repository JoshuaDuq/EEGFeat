(function () {
  document.querySelectorAll("dl.py.class > dd > p:first-child").forEach(function (paragraph) {
    var first = paragraph.firstChild;
    var text = first && first.nodeType === Node.TEXT_NODE ? first.textContent : "";
    if (text.trim().indexOf("Bases:") === 0) paragraph.classList.add("api-bases");
  });

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
