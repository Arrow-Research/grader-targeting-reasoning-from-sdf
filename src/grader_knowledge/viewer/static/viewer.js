// Rollout viewer (viewer package): show/hide toggles and n / p keyboard navigation between episodes.
// show/hide toggles: <button class="toggle" data-target="id">show|hide</button>
document.addEventListener("click", (event) => {
  const button = event.target.closest("button.toggle[data-target]");
  if (!button) return;
  const target = document.getElementById(button.dataset.target);
  if (!target) return;
  const nowHidden = target.classList.toggle("hidden");
  if (button.textContent === "show" || button.textContent === "hide") {
    button.textContent = nowHidden ? "show" : "hide";
  }
});

// keyboard: n / p = next / previous episode (episode page)
document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  if (event.key !== "n" && event.key !== "p") return;
  const link = document.querySelector(`.pager a[rel='${event.key === "n" ? "next" : "prev"}']`);
  if (link) window.location = link.href;
});
