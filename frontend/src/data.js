import Papa from 'papaparse'

const FRAME_COUNT = 100
const FRAME_INTERVAL_MIN = 14.4

export async function loadDatasets() {
  const resp = await fetch('/api/datasets')
  return resp.json()
}

export async function loadDatasetSummary(datasetId) {
  const resp = await fetch(`/data/${datasetId}/summary.json`)
  if (!resp.ok) return null
  return resp.json()
}

export async function loadDatasetData(datasetId) {
  try {
    const [measResp, summResp] = await Promise.all([
      fetch(`/data/${datasetId}/structure_measurements.csv`),
      fetch(`/data/${datasetId}/track_summary.csv`),
    ])
    if (!measResp.ok || !summResp.ok) return { measurements: [], trackSummary: [] }
    const measText = await measResp.text()
    const summText = await summResp.text()
    const measurements = Papa.parse(measText, { header: true, dynamicTyping: true }).data
      .filter(r => r.frame != null)
    const trackSummary = Papa.parse(summText, { header: true, dynamicTyping: true }).data
      .filter(r => r.track_id != null)
    return { measurements, trackSummary }
  } catch {
    return { measurements: [], trackSummary: [] }
  }
}

/**
 * @param {'raw'|'mask'|'diagnostic'} mode
 */
export function getFrameUrl(datasetId, frameIdx, mode = 'raw') {
  if (mode === true) mode = 'mask'
  else if (mode === false) mode = 'raw'
  const folder = mode === 'mask' ? 'masks' : mode === 'diagnostic' ? 'diagnostics' : 'raw'
  return `/data/${datasetId}/${folder}/frame_${String(frameIdx).padStart(3, '0')}.png`
}

export function frameToTimestamp(frame) {
  const totalMin = frame * FRAME_INTERVAL_MIN
  const hours = Math.floor(totalMin / 60)
  const minutes = Math.round(totalMin % 60)
  return `${hours}h ${minutes}m`
}

const TRACK_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
  '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]

export function getTrackColor(trackId) {
  return TRACK_COLORS[(trackId - 1) % TRACK_COLORS.length]
}

export async function uploadTifFile(file) {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch('/api/upload', { method: 'POST', body: form })
  return resp.json()
}

export async function runPipeline(datasetId, params) {
  const resp = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, params }),
  })
  return resp.json()
}

export async function deleteDataset(datasetId) {
  const resp = await fetch(`/api/dataset/${datasetId}`, { method: 'DELETE' })
  return resp.json()
}

export async function saveRunName(datasetId, runName) {
  const resp = await fetch(`/api/dataset/${datasetId}/run-name`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_name: runName }),
  })
  return resp.json()
}

export async function deleteResults(datasetId) {
  const resp = await fetch(`/api/dataset/${datasetId}/results`, { method: 'DELETE' })
  return resp.json()
}

export async function pollJobStatus(jobId) {
  const resp = await fetch(`/api/status/${jobId}`)
  return resp.json()
}

export { FRAME_COUNT, FRAME_INTERVAL_MIN }
