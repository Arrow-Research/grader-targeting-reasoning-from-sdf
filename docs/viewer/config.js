// Where the static rollout viewer reads its data (docs/viewer/README.md). Both are base URLs under which the dataset
// repos' files are fetched by their repo paths (index.json, rollouts/..., <universe>/universe.json, ...).
// Override per page load with ?data=<url>&corpora=<url> (e.g. a local copy served by python3 -m http.server).
window.VIEWER_CONFIG = {
  data: "https://huggingface.co/datasets/arrow-research/grader-targeting-rollouts/resolve/main",
  corpora: "https://huggingface.co/datasets/arrow-research/grader-targeting-sdf-corpora/resolve/main",
};
