(() => {
  document.querySelectorAll("dl.py.class > dd > p:first-child").forEach((paragraph) => {
    if (paragraph.textContent.trim().startsWith("Bases:")) {
      paragraph.classList.add("api-bases");
    }
  });
  document.querySelectorAll(".sidebar-tree .current-page > a").forEach((link) => {
    link.setAttribute("aria-current", "page");
  });

  const scrollers = document.querySelectorAll(
    "article .table-wrapper, article div.highlight pre, article .sig:not(.sig-inline), article div.math",
  );
  const markOverflow = (element) => {
    const overflowing = element.scrollWidth > element.clientWidth + 1;
    element.classList.toggle("is-overflowing", overflowing);
    element.classList.toggle("at-start", element.scrollLeft <= 1);
    element.classList.toggle("at-end", element.scrollLeft + element.clientWidth >= element.scrollWidth - 1);
    if (overflowing) element.tabIndex = 0;
    else element.removeAttribute("tabindex");
  };
  const resizeObserver = new ResizeObserver((entries) => {
    entries.forEach(({ target }) => markOverflow(target));
  });
  scrollers.forEach((element) => {
    resizeObserver.observe(element);
    element.addEventListener("scroll", () => markOverflow(element), { passive: true });
    if (element.matches(".table-wrapper")) {
      element.setAttribute("role", "region");
      element.setAttribute("aria-label", "Data table. Scroll horizontally to view additional columns.");
    }
  });
  document.fonts.ready.then(() => scrollers.forEach(markOverflow));

  function revealLinkedTab() {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target && target.matches(".sd-tab-label")) target.click();
  }
  window.addEventListener("hashchange", revealLinkedTab);
  revealLinkedTab();
})();
