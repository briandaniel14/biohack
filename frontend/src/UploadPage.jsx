import { useState, useCallback } from 'react'
import { uploadTifFile, pollJobStatus, loadDatasets, deleteDataset, deleteResults } from './data.js'

export default function UploadPage({
  datasets,
  currentDataset,
  onSelectDataset,
  onDatasetsChange,
  onNavigateTuning,
  onNavigateResults,
  pipelineJobs,
  pipelineRunning,
  pipelineDatasetId,
  pipelineStep,
}) {
  const [uploadStatus, setUploadStatus] = useState(null)
  const [dragOver, setDragOver] = useState(false)

  const handleUpload = useCallback(async (file) => {
    if (!file) return
    const ext = file.name.split('.').pop().toLowerCase()
    if (ext !== 'tif' && ext !== 'tiff') {
      setUploadStatus({ state: 'error', message: 'Only .tif/.tiff files accepted' })
      return
    }
    setUploadStatus({ state: 'uploading', message: `Uploading ${file.name}...` })
    try {
      const { job_id, error } = await uploadTifFile(file)
      if (error) { setUploadStatus({ state: 'error', message: error }); return }
      setUploadStatus({ state: 'processing', message: 'Splitting frames...' })
      const poll = setInterval(async () => {
        const status = await pollJobStatus(job_id)
        if (status.status === 'complete') {
          clearInterval(poll)
          setUploadStatus({ state: 'done', message: `Done — ${status.dataset_id}` })
          const ds = await loadDatasets()
          onDatasetsChange(ds)
          const newDs = ds.find(d => d.id === status.dataset_id)
          if (newDs) onSelectDataset(newDs)
          setTimeout(() => setUploadStatus(null), 3000)
        } else if (status.status === 'error') {
          clearInterval(poll)
          setUploadStatus({ state: 'error', message: status.error || 'Processing failed' })
        } else {
          setUploadStatus({ state: 'processing', message: status.step || 'Processing...' })
        }
      }, 2000)
    } catch (e) {
      setUploadStatus({ state: 'error', message: `Upload failed: ${e.message}` })
    }
  }, [onDatasetsChange, onSelectDataset])

  const handleDelete = useCallback(async (e, dsId) => {
    e.stopPropagation()
    await deleteDataset(dsId)
    const ds = await loadDatasets()
    onDatasetsChange(ds)
    if (currentDataset?.id === dsId) {
      onSelectDataset(ds.length > 0 ? ds[0] : null)
    }
  }, [currentDataset, onDatasetsChange, onSelectDataset])

  const handleDeleteRun = useCallback(async (e, dsId) => {
    e.stopPropagation()
    await deleteResults(dsId)
    const ds = await loadDatasets()
    onDatasetsChange(ds)
  }, [onDatasetsChange])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    if (file) handleUpload(file)
  }, [handleUpload])

  const handleFileInput = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) handleUpload(file)
    e.target.value = ''
  }, [handleUpload])

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Top: Upload — full width */}
      <div className="flex-none p-4 sm:p-6 pb-0">
        <div className="max-w-5xl mx-auto rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
          <div className="px-6 py-3.5 border-b border-gray-800 flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-gray-100">Upload New Dataset</h2>
              <p className="text-xs text-gray-500 mt-0.5">Multi-frame grayscale TIFF stack</p>
            </div>
          </div>
          <div className="p-5">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[120px] ${
                dragOver
                  ? 'border-blue-400 bg-blue-400/10'
                  : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/40'
              }`}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('tif-upload').click()}
            >
              <input
                id="tif-upload"
                type="file"
                accept=".tif,.tiff"
                className="hidden"
                onChange={handleFileInput}
              />
              <div className="w-11 h-11 rounded-full bg-gray-800 flex items-center justify-center mb-2.5">
                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4" />
                </svg>
              </div>
              <div className="min-h-[44px] flex flex-col items-center justify-center">
                {uploadStatus ? (
                  <div className={`text-sm font-medium ${
                    uploadStatus.state === 'error' ? 'text-red-400' :
                    uploadStatus.state === 'done' ? 'text-green-400' :
                    'text-green-400'
                  }`}>
                    {uploadStatus.state === 'processing' && <svg className="inline-block w-4 h-4 animate-spin mr-1.5" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>}
                    {uploadStatus.message}
                  </div>
                ) : (
                  <>
                    <p className="text-sm text-gray-300">
                      Drop .tif/.tiff here or <span className="text-blue-400 hover:text-blue-400 underline underline-offset-2">browse</span>
                    </p>
                    <p className="text-xs text-gray-600 mt-1">Accepted formats: .tif, .tiff</p>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: Two-column split — Files + Runs */}
      <div className="flex-1 min-h-0 p-4 sm:p-6">
        <div className="max-w-5xl mx-auto h-full flex flex-col md:flex-row gap-4 md:gap-6">

          {/* Left: Previous Files */}
          <div className="flex-1 min-w-0 flex flex-col rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
            <div className="flex-none px-5 py-3.5 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                <h2 className="text-sm font-semibold text-gray-100">Image Files</h2>
              </div>
              <p className="text-[11px] text-gray-500 mt-0.5">Uploaded TIF stacks</p>
            </div>
            <div className="flex-1 min-h-0 overflow-auto divide-y divide-gray-800/60">
              {datasets.length === 0 ? (
                <div className="px-5 py-10 text-center text-sm text-gray-600">No files uploaded yet</div>
              ) : datasets.map(ds => (
                <div
                  key={ds.id}
                  onClick={() => onSelectDataset(ds)}
                  className={`w-full text-left px-5 py-3 flex items-center gap-3 transition-colors hover:bg-gray-800/50 cursor-pointer ${
                    currentDataset?.id === ds.id ? 'bg-blue-400/15 border-l-2 border-blue-400' : ''
                  }`}
                >
                  <div className={`w-2 h-2 rounded-full flex-none ${currentDataset?.id === ds.id ? 'bg-blue-400' : 'bg-gray-700'}`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm truncate ${currentDataset?.id === ds.id ? 'text-blue-400 font-medium' : 'text-gray-300'}`}>
                      {ds.name}
                    </p>
                    <p className="text-[11px] text-gray-600 mt-0.5">{ds.frames} frames · {ds.timeSpan}</p>
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, ds.id)}
                    className="p-1 rounded-md text-gray-600 hover:text-red-400 hover:bg-red-500/10 transition-colors flex-none"
                    title="Delete dataset"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Previous Runs */}
          <div className="flex-1 min-w-0 flex flex-col rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
            <div className="flex-none px-5 py-3.5 border-b border-gray-800">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h2 className="text-sm font-semibold text-gray-100">Pipeline Runs</h2>
              </div>
              <p className="text-[11px] text-gray-500 mt-0.5">Detection results per dataset</p>
            </div>
            <div className="flex-1 min-h-0 overflow-auto divide-y divide-gray-800/60">
              {/* Running jobs */}
              {pipelineJobs && Object.entries(pipelineJobs).filter(([, j]) => j.running).map(([jid, job]) => {
                const runDs = datasets.find(d => d.id === job.datasetId)
                return (
                  <div key={jid} className="w-full text-left px-5 py-3 flex items-center gap-3 bg-gray-800/30 animate-pulse">
                    <div className="w-7 h-7 rounded-lg bg-amber-400/10 flex items-center justify-center flex-none">
                      <svg className="w-3.5 h-3.5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" strokeWidth={1.5} />
                        <path strokeLinecap="round" strokeWidth={2} d="M12 6v6l4 2" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-amber-300 truncate">{runDs?.name || job.datasetId}</p>
                      <p className="text-[11px] text-amber-400/70 mt-0.5">{job.step || 'Processing...'}</p>
                    </div>
                  </div>
                )
              })}

              {/* Completed runs */}
              {(!pipelineJobs || !Object.values(pipelineJobs).some(j => j.running)) && datasets.filter(ds => ds.has_results).length === 0 && (
                <div className="px-5 py-10 text-center text-sm text-gray-600">No runs yet</div>
              )}
              {datasets.filter(ds => ds.has_results).map(ds => (
                <div
                  key={ds.id}
                  onClick={() => { onSelectDataset(ds); onNavigateResults() }}
                  className="w-full text-left px-5 py-3 flex items-center gap-3 transition-colors hover:bg-gray-800/50 cursor-pointer"
                >
                  <div className="w-7 h-7 rounded-lg bg-green-400/10 flex items-center justify-center flex-none">
                    <svg className="w-3.5 h-3.5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-300 truncate">{ds.run_name || ds.name}</p>
                    <p className="text-[11px] text-gray-600 mt-0.5">{ds.name}{ds.completed_at ? ` · ${new Date(ds.completed_at).toLocaleString()}` : ''}</p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteRun(e, ds.id)}
                    className="p-1 rounded-md text-gray-600 hover:text-red-400 hover:bg-red-500/10 transition-colors flex-none"
                    title="Delete run results"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="flex-none p-4 sm:p-6 pt-0">
        <div className="max-w-5xl mx-auto">
          <button
            onClick={onNavigateTuning}
            disabled={!currentDataset}
            className="w-full py-3 rounded-xl bg-blue-700 hover:bg-blue-600 text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            Continue to Tuning
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5-5 5M6 12h12" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
