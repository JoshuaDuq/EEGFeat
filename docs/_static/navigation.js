(() => {
  const content = document.querySelector(".content");
  const headers = document.querySelectorAll(".docs-header, .mobile-header");
  const search = document.querySelector(".sidebar-search");
  const headerSearch = document.querySelector("#header-search-query");
  const drawers = [
    {
      toggle: document.getElementById("__navigation"),
      panel: document.querySelector(".sidebar-drawer"),
      name: "Documentation navigation",
    },
    {
      toggle: document.getElementById("__toc"),
      panel: document.querySelector(".toc-drawer"),
      name: "On this page",
    },
  ];
  let activeDrawer = null;

  const isOverlay = (drawer) => getComputedStyle(drawer.panel).position === "fixed";
  const visibleControl = (drawer) => drawer.controls.find((control) => control.getClientRects().length);

  function synchronizeDrawers() {
    activeDrawer = drawers.find((drawer) => drawer.toggle.checked && isOverlay(drawer));
    content.inert = Boolean(activeDrawer);
    headers.forEach((header) => { header.inert = Boolean(activeDrawer); });
    document.body.classList.toggle("drawer-open", Boolean(activeDrawer));
    drawers.forEach((drawer) => {
      const overlay = isOverlay(drawer);
      drawer.panel.dataset.overlay = String(overlay);
      drawer.panel.inert = (overlay && !drawer.toggle.checked)
        || Boolean(activeDrawer && activeDrawer !== drawer);
      drawer.controls.forEach((control) => {
        control.setAttribute("aria-expanded", String(drawer.toggle.checked));
      });
      if (activeDrawer === drawer) {
        drawer.panel.setAttribute("role", "dialog");
        drawer.panel.setAttribute("aria-modal", "true");
      } else {
        drawer.panel.removeAttribute("role");
        drawer.panel.removeAttribute("aria-modal");
      }
    });
  }

  function closeDrawer(drawer) {
    drawer.toggle.checked = false;
    synchronizeDrawers();
    drawer.trigger.focus({ preventScroll: true });
  }

  function openSearch(trigger) {
    if (headerSearch.getClientRects().length) {
      headerSearch.focus();
      return;
    }
    const navigation = drawers[0];
    if (isOverlay(navigation)) {
      navigation.trigger = trigger;
      navigation.toggle.checked = true;
      navigation.toggle.dispatchEvent(new Event("change"));
    }
    search.focus();
  }

  const mobileSearch = document.createElement("button");
  mobileSearch.type = "button";
  mobileSearch.className = "mobile-search";
  mobileSearch.setAttribute("aria-label", "Search documentation");
  mobileSearch.innerHTML = '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    + '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></svg>';
  mobileSearch.addEventListener("click", () => openSearch(mobileSearch));
  document.querySelector(".mobile-header .header-right").prepend(mobileSearch);

  const searchShortcut = document.querySelector(".header-search kbd");
  searchShortcut.textContent = navigator.platform.startsWith("Mac") ? "⌘ K" : "Ctrl K";

  drawers.forEach((drawer) => {
    drawer.panel.id = drawer.toggle.id === "__navigation" ? "site-navigation" : "page-contents";
    drawer.panel.setAttribute("aria-label", drawer.name);
    drawer.controls = Array.from(document.querySelectorAll(`label[for="${drawer.toggle.id}"]:not(.overlay):not(.no-toc)`));
    drawer.controls.forEach((control) => {
      control.tabIndex = 0;
      control.setAttribute("role", "button");
      control.setAttribute("aria-label", drawer.toggle.getAttribute("aria-label"));
      control.setAttribute("aria-controls", drawer.panel.id);
      control.addEventListener("click", () => { drawer.trigger = control; });
      control.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          control.click();
        }
      });
    });

    const tools = document.createElement("div");
    tools.className = "drawer-tools";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "drawer-close";
    close.setAttribute("aria-label", `Close ${drawer.name.toLowerCase()}`);
    close.textContent = "Close ×";
    close.addEventListener("click", () => closeDrawer(drawer));
    tools.append(close);
    drawer.panel.prepend(tools);

    drawer.toggle.addEventListener("change", () => {
      if (drawer.toggle.checked) {
        drawers.forEach((other) => {
          if (other !== drawer) other.toggle.checked = false;
        });
        synchronizeDrawers();
        if (isOverlay(drawer)) {
          if (drawer === drawers[0]) search.focus();
          else close.focus();
        }
      } else {
        closeDrawer(drawer);
      }
    });
    drawer.panel.addEventListener("click", (event) => {
      if (event.target.closest('a[href^="#"]') && activeDrawer === drawer) closeDrawer(drawer);
    });
  });

  document.querySelectorAll(".sidebar-tree label[for]").forEach((control) => {
    const toggle = document.getElementById(control.htmlFor);
    const title = control.parentElement.querySelector("a").textContent;
    control.tabIndex = 0;
    control.setAttribute("role", "button");
    const update = () => {
      control.setAttribute("aria-expanded", String(toggle.checked));
      control.setAttribute("aria-label", `${toggle.checked ? "Collapse" : "Expand"} ${title}`);
    };
    toggle.addEventListener("change", update);
    update();
    control.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        control.click();
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && activeDrawer) {
      event.preventDefault();
      closeDrawer(activeDrawer);
      return;
    }
    if (event.key === "Tab" && activeDrawer) {
      const focusable = Array.from(activeDrawer.panel.querySelectorAll('a[href], button, input, [tabindex="0"]'))
        .filter((element) => element.getClientRects().length && !element.disabled);
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
      return;
    }
    const isEditing = event.target.closest("input, textarea, select, [contenteditable]");
    const slash = event.key === "/" && !isEditing && !event.metaKey && !event.ctrlKey && !event.altKey;
    const shortcut = (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k";
    if (!slash && !shortcut) return;
    event.preventDefault();
    const trigger = document.activeElement === document.body
      ? visibleControl(drawers[0])
      : document.activeElement;
    openSearch(trigger);
  });
  window.addEventListener("resize", synchronizeDrawers);
  synchronizeDrawers();
})();
