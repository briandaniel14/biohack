import { useState, useCallback } from 'react'
import { getFrameUrl, uploadTifFile, pollJobStatus, loadDatasets } from './data.js'
import HyperparamPanel from './HyperparamPanel.jsx'

export default function ProcessPage({
  datasets,
  currentDataset,
  onSelectDataset,
  onDatasetsChange,
  onNavigateResults,
}) {
  const [uploadStatus, setUploadStatus] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [previewFrame, setPreviewFrame] = useState(0)
  const [showAnnotated, setShowAnnotated] = useState(false)
  const [processing, setProcessing] = useState(false)

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
      if (error) {
        setUploadStatus({ state: 'error', message: error })
        return
      }
      setUploadStatus({ state: 'processing', message: 'Running detection pipeline...' })
      setProcessing(true)
      const poll = setInterval(async () => {
        const status = await pollJobStatus(job_id)
        if (status.status === 'complete') {
          clearInterval(poll)
          setUploadStatus({ state: 'done', message: `Done — ${status.dataset_id}` })
          setProcessing(false)
          const ds = await loadDatasets()
          onDatasetsChange(ds)
          const newDs = ds.find(d => d.id === status.dataset_id)
          if (newDs) onSelectDataset(newDs)
          setShowAnnotated(true)
        } else if (status.status === 'error') {
          clearInterval(poll)
          setUploadStatus({ state: 'error', message: status.error || 'Processing failed' })
          setProcessing(false)
        } else {
          setUploadStatus({ state: 'processing', message: status.step || 'Processing...' })
        }
      }, 2000)
    } catch (e) {
      setUploadStatus({ state: 'error', message: `Upload failed: ${e.message}` })
      setProcessing(false)
    }
  }, [onDatasetsChange, onSelectDataset])

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
    <div className="flex-1 flex min-h-0">
      {/* LEFT: Upload + Image Preview */}
      <div className="w-1/2 p-6 flex flex-col gap-4 border-r border-gray-800">
        {/* Upload area */}
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors flex flex-col items-center justify-center min-h-[140px] ${
            dragOver ? 'border-blue-400 bg-blue-900/20' : 'border-gray-700 hover:border-gray-500'
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
          <svg className="w-10 h-10 text-gray-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4" />
          </svg>
          {uploadStatus ? (
            <div className={`text-sm ${
              uploadStatus.state === 'error' ? 'text-red-400' :
              uploadStatus.state === 'done' ? 'text-green-400' :
              'text-green-400'
            }`}>
              {uploadStatus.state === 'processing' && <span className="inline-block animate-spin mr-1">⟳</span>}
              {uploadStatus.message}
            </div>
          ) : (
            <div>
              <div className="text-sm text-gray-400">
                Drop .tif/.tiff here or <span className="text-blue-400 underline">click to upload</span>
              </div>
              <div className="text-xs text-gray-600 mt-1">Multi-frame TIFF stack</div>
            </div>
          )}
        </div>

        {/* Dataset selector */}
        {datasets.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-400">Dataset:</span>
            <select
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm"
              value={currentDataset?.id || ''}
              onChange={e => {
                const ds = datasets.find(d => d.id === e.target.value)
                if (ds) { onSelectDataset(ds); setShowAnnotated(false) }
              }}
            >
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.name} ({ds.id})</option>
              ))}
            </select>
          </div>
        )}

        {/* Image preview */}
        {currentDataset && (
          <div className="flex-1 min-h-0 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Preview — Frame {previewFrame + 1}</span>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showAnnotated}
                  onChange={e => setShowAnnotated(e.target.checked)}
                  className="accent-blue-500"
                />
                <span className="text-xs">Show detections</span>
              </label>
            </div>
            <div className="relative bg-black flex-1 min-h-0 flex items-center justify-center">
              <img
                src={getFrameUrl(currentDataset.id, previewFrame, showAnnotated)}
                alt={`Frame ${previewFrame + 1}`}
                className="w-full max-h-full"
                style={{ imageRendering: 'pixelated', aspectRatio: '1 / 1' }}
              />
            </div>
            <input
              type="range"
              min={0}
              max={(currentDataset?.frames || 1) - 1}
              value={previewFrame}
              onChange={e => setPreviewFrame(parseInt(e.target.value))}
              className="w-full accent-blue-500"
            />
            {/* View Results button */}
            <button
              onClick={onNavigateResults}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed"
              disabled={!currentDataset}
            >
              View Results →
            </button>
          </div>
        )}
      </div>

      {/* RIGHT: Hyperparameters + Process */}
      <div className="w-1/2 flex flex-col min-h-0">
        <div className="flex-1 min-h-0">
          <HyperparamPanel />
        </div>
        <div className="flex-none p-4 border-t border-gray-800">
          <button
            onClick={() => {/* TODO: POST hyperparams to backend and re-run pipeline */}}
            disabled={!currentDataset || processing}
            className="w-full py-3 bg-blue-700 hover:bg-blue-600 rounded-lg text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {processing ? (
              <><span className="inline-block animate-spin">⟳</span> Processing...</>
            ) : (
              <><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Run Pipeline</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
