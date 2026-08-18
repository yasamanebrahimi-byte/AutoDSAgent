const viewTitles = {
  overview: "Run overview",
  workflow: "Workflow trace & approval",
  reports: "Generated report preview",
  artifacts: "Artifact browser",
};

const workflowOrder = ["profile", "cleaning", "approval", "eda", "modeling", "report"];

const tourSteps = [
  {
    delay: 0,
    title: "Creating a reproducible run",
    copy: "The uploaded CSV is preserved before any derived artifact is written.",
    view: "overview",
    status: "Preparing run",
    progress: 5,
    active: "profile",
    completeThrough: -1,
  },
  {
    delay: 5000,
    title: "Profiling schema and quality",
    copy: "Column roles, missingness, duplicates, and leakage signals are persisted.",
    view: "overview",
    status: "Running · profile",
    progress: 18,
    active: "profile",
    completeThrough: -1,
  },
  {
    delay: 10000,
    title: "Planning conservative cleaning",
    copy: "Structural actions are separated from learned preprocessing.",
    view: "workflow",
    status: "Running · cleaning",
    progress: 34,
    active: "cleaning",
    completeThrough: 0,
  },
  {
    delay: 15000,
    title: "Waiting at the human approval gate",
    copy: "Risky or expensive work pauses without losing workflow state.",
    view: "workflow",
    status: "Waiting for approval",
    progress: 43,
    active: "approval",
    completeThrough: 1,
    waiting: true,
  },
  {
    delay: 21000,
    title: "Approval recorded; analysis resumed",
    copy: "The decision and generation are appended to the audit trace.",
    view: "workflow",
    status: "Running · EDA",
    progress: 58,
    active: "eda",
    completeThrough: 2,
  },
  {
    delay: 27000,
    title: "Selecting on training-only cross-validation",
    copy: "One selected pipeline is evaluated once on the untouched holdout.",
    view: "overview",
    status: "Running · modeling",
    progress: 78,
    active: "modeling",
    completeThrough: 3,
  },
  {
    delay: 33000,
    title: "Rendering from terminal workflow state",
    copy: "The final report links metrics, limitations, lineage, and source artifacts.",
    view: "reports",
    status: "Running · report",
    progress: 92,
    active: "report",
    completeThrough: 4,
  },
  {
    delay: 39000,
    title: "Run completed with 24 inspectable artifacts",
    copy: "Models, reports, plots, state, and traces remain downloadable and linked.",
    view: "artifacts",
    status: "Completed",
    progress: 100,
    active: null,
    completeThrough: 5,
  },
];

let tourTimers = [];
let tourRunning = false;

function setView(view) {
  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === view;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  document.querySelectorAll("[data-view-button]").forEach((button) => {
    button.setAttribute("aria-selected", button.dataset.viewButton === view ? "true" : "false");
  });
  document.querySelector("#workspace-title").textContent = viewTitles[view];
}

function setWorkflowState(step) {
  const completeThrough = step.completeThrough ?? 5;
  workflowOrder.forEach((name, index) => {
    const item = document.querySelector(`[data-workflow-step="${name}"]`);
    item.classList.toggle("complete", index <= completeThrough);
    item.classList.toggle("active", name === step.active);
    item.classList.toggle("waiting", Boolean(step.waiting) && name === step.active);
    const marker = item.querySelector(".step-state");
    if (index <= completeThrough) marker.textContent = "✓";
    else if (name === step.active && step.waiting) marker.textContent = "!";
    else if (name === step.active) marker.textContent = "•";
    else marker.textContent = String(index + 1);
  });

  const approvalCard = document.querySelector("#approval-card");
  const approvalButton = document.querySelector("#approve-demo");
  const waiting = Boolean(step.waiting);
  approvalCard.classList.toggle("waiting", waiting);
  approvalCard.querySelector(".approval-icon").textContent = waiting ? "!" : "✓";
  document.querySelector("#approval-title").textContent = waiting
    ? "Modeling approval required"
    : "Approval policy acknowledged";
  document.querySelector("#approval-copy").textContent = waiting
    ? "The workflow is safely paused before model training. Review the target, holdout policy, and estimated work."
    : "Risky cleaning and modeling can pause the workflow before execution. Every decision is recorded in the trace.";
  document.querySelector("#approval-decision").textContent = waiting ? "Pending" : "Approved";
  approvalButton.textContent = waiting ? "Approve & resume" : "Approved";
  approvalButton.disabled = !waiting;
}

function renderTourStep(step, index) {
  setView(step.view);
  setWorkflowState(step);
  document.querySelector("#tour-step-number").textContent = String(index + 1);
  document.querySelector("#tour-step-title").textContent = step.title;
  document.querySelector("#tour-step-copy").textContent = step.copy;
  document.querySelector("#run-status").textContent = step.status;
  document.querySelector("#run-progress").style.width = `${step.progress}%`;
  const completeCount = Math.max(0, step.completeThrough + 1);
  document.querySelector("#run-progress-label").textContent = `${completeCount} / 6 stages`;
}

function stopTour({ reset = false } = {}) {
  tourTimers.forEach(window.clearTimeout);
  tourTimers = [];
  tourRunning = false;
  document.body.classList.remove("tour-running");
  document.querySelector("#tour-banner").hidden = true;
  document.querySelector("#replay-tour").innerHTML = '<span aria-hidden="true">▶</span> Replay 40-second tour';
  if (reset) {
    setView("overview");
    setWorkflowState({ completeThrough: 5, active: null });
    document.querySelector("#run-status").textContent = "Completed";
    document.querySelector("#run-progress").style.width = "100%";
    document.querySelector("#run-progress-label").textContent = "6 / 6 stages";
  }
}

function startTour() {
  stopTour();
  tourRunning = true;
  document.body.classList.add("tour-running");
  document.querySelector("#tour-banner").hidden = false;
  document.querySelector("#replay-tour").textContent = "Tour playing…";

  tourSteps.forEach((step, index) => {
    tourTimers.push(window.setTimeout(() => {
      renderTourStep(step, index);
      if (index === tourSteps.length - 1) {
        tourTimers.push(window.setTimeout(() => stopTour(), 3500));
      }
    }, step.delay));
  });
}

document.querySelectorAll("[data-view-button]").forEach((button) => {
  button.addEventListener("click", () => {
    if (tourRunning) stopTour();
    setView(button.dataset.viewButton);
  });
});

document.querySelectorAll("[data-view-link]").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.viewLink));
});

document.querySelector("#replay-tour").addEventListener("click", startTour);
document.querySelector("#stop-tour").addEventListener("click", () => stopTour({ reset: true }));
document.querySelector("#approve-demo").addEventListener("click", () => {
  if (!tourRunning) return;
  const resumeStep = tourSteps[4];
  renderTourStep(resumeStep, 4);
});

setView("overview");
setWorkflowState({ completeThrough: 5, active: null });
