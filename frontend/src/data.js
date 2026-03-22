import Papa from 'papaparse'

const FRAME_INTERVAL_MIN = 15
const PX_TO_UM = 0.3

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
    const resp = await fetch(`/data/${datasetId}/cell_tracks.csv`)
    if (!resp.ok) return { rows: [], filamentSummary: [] }
    const text = await resp.text()
    const rows = Papa.parse(text, { header: true, dynamicTyping: true }).data
      .filter(r => r.frame != null)

    // Normalize frames to 0-based (new pipeline uses 1-based frames)
    const minFrame = rows.length > 0 ? Math.min(...rows.map(r => r.frame)) : 0
    const frameOffset = minFrame >= 1 ? minFrame : 0
    if (frameOffset > 0) {
      for (const r of rows) {
        r.frame -= frameOffset
        if (r.time_of_appearance != null) r.time_of_appearance -= frameOffset
      }
    }

    // Derive filament summary (one entry per filament_ID)
    const filMap = new Map()
    for (const r of rows) {
      if (r.filament_present !== 1 || r.filament_ID == null || r.filament_ID === '') continue
      const fid = r.filament_ID
      if (!filMap.has(fid)) {
        filMap.set(fid, {
          filament_ID: fid,
          host_cell_ID: r.cell_ID,
          first_frame: r.frame,
          last_frame: r.frame,
          frame_count: 0,
          time_of_appearance: r.time_of_appearance,
          _lengths: [],
          _areas: [],
          _eccs: [],
        })
      }
      const f = filMap.get(fid)
      f.last_frame = Math.max(f.last_frame, r.frame)
      f.first_frame = Math.min(f.first_frame, r.frame)
      f.frame_count++
      if (r.filament_major_axis_length != null) f._lengths.push(r.filament_major_axis_length)
      if (r.filament_area != null) f._areas.push(r.filament_area)
      if (r.filament_eccentricity != null) f._eccs.push(r.filament_eccentricity)
    }
    const filamentSummary = [...filMap.values()].map(f => {
      f.avg_length = f._lengths.length > 0
        ? +(f._lengths.reduce((a, b) => a + b, 0) / f._lengths.length * PX_TO_UM).toFixed(2)
        : 0
      f.max_length = f._lengths.length > 0 ? +(Math.max(...f._lengths) * PX_TO_UM).toFixed(2) : 0
      f.avg_area = f._areas.length > 0
        ? +(f._areas.reduce((a, b) => a + b, 0) / f._areas.length * PX_TO_UM * PX_TO_UM).toFixed(2)
        : 0
      f.avg_eccentricity = f._eccs.length > 0
        ? +(f._eccs.reduce((a, b) => a + b, 0) / f._eccs.length).toFixed(2)
        : 0
      delete f._lengths; delete f._areas; delete f._eccs
      return f
    }).sort((a, b) => a.filament_ID - b.filament_ID)

    console.log('[loadDatasetData]', datasetId, 'rows:', rows.length, 'filaments:', filamentSummary.length, 'sample:', rows[0])
    return { rows, filamentSummary }
  } catch (e) {
    console.error('loadDatasetData error:', e)
    return { rows: [], filamentSummary: [] }
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

export function getTrackColor(id) {
  return TRACK_COLORS[((id || 1) - 1) % TRACK_COLORS.length]
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
  if (!resp.ok) return { status: 'error', error: 'Job not found (server may have restarted)' }
  return resp.json()
}

export async function fetchActiveJobs() {
  const resp = await fetch('/api/jobs')
  if (!resp.ok) return {}
  return resp.json()
}

export { FRAME_INTERVAL_MIN, PX_TO_UM }
