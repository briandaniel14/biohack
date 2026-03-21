import { useState, useEffect, useRef, useCallback } from 'react'
import { loadDatasets, loadDatasetSummary, runPipeline, pollJobStatus } from './data.js'
import UploadPage from './UploadPage.jsx'
import TuningPage from './TuningPage.jsx'
import ResultsPage from './ResultsPage.jsx'

const PAGES = [
  { key: 'upload', label: 'Upload', icon: 'M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4' },
  { key: 'tuning', label: 'Tuning', icon: 'M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4' },
  { key: 'results', label: 'Results', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
]

export default function App() {
  const [page, setPage] = useState('upload')
  const [datasets, setDatasets] = useState([])
  const [currentDataset, setCurrentDataset] = useState(null)

  // Pipeline job state — lives at App level so it persists across page nav
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineStep, setPipelineStep] = useState('')
  const [pipelineDatasetId, setPipelineDatasetId] = useState(null)
  const pollRef = useRef(null)

  const navigateTo = (p) => {
    setPage(p)
    if (p === 'upload') {
      loadDatasets().then(ds => {
        setDatasets(ds)
        if (!currentDataset && ds.length > 0) setCurrentDataset(ds[0])
      }).catch(() => {})
    }
  }

  useEffect(() => {
    loadDatasets().then(ds => {
      setDatasets(ds)
      if (ds.length > 0) setCurrentDataset(ds[0])
    }).catch(() => {})
  }, [])

  // Cleanup poll on unmount
  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleRunPipeline = useCallback(async (datasetId, params) => {
    if (pipelineRunning) return
    setPipelineRunning(true)
    setPipelineStep('Starting...')
    setPipelineDatasetId(datasetId)
    try {
      const { job_id, error } = await runPipeline(datasetId, params)
      if (error) { setPipelineStep(`Error: ${error}`); setPipelineRunning(false); return }

      pollRef.current = setInterval(async () => {
        const status = await pollJobStatus(job_id)
        if (status.status === 'complete') {
          clearInterval(pollRef.current)
          pollRef.current = null
          setPipelineStep('Done!')
          setPipelineRunning(false)
          setPipelineDatasetId(null)
          // Refresh datasets list so has_results updates
          loadDatasets().then(ds => setDatasets(ds)).catch(() => {})
          setTimeout(() => setPipelineStep(''), 5000)
        } else if (status.status === 'error') {
          clearInterval(pollRef.current)
          pollRef.current = null
          setPipelineStep(`Error: ${status.error || 'Pipeline failed'}`)
          setPipelineRunning(false)
          setPipelineDatasetId(null)
        } else {
          setPipelineStep(status.step || 'Processing...')
        }
      }, 2000)
    } catch (e) {
      setPipelineStep(`Error: ${e.message}`)
      setPipelineRunning(false)
      setPipelineDatasetId(null)
    }
  }, [pipelineRunning])

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="flex-none px-5 py-2.5 bg-gray-900 border-b border-gray-800 flex items-center gap-4">
        <img src="/logo.png?v=2" alt="Logo" className="w-9 h-9" />
        <h1 className="text-base font-semibold text-white tracking-tight">Filament Detection</h1>
        <div className="h-5 w-px bg-gray-800 mx-1" />
        <nav className="flex gap-0.5">
          {PAGES.map((p, i) => (
            <button
              key={p.key}
              onClick={() => navigateTo(p.key)}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-sm font-medium transition-all ${
                page === p.key
                  ? 'bg-blue-700 text-white shadow-sm shadow-blue-700/25'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d={p.icon} />
              </svg>
              {p.label}
            </button>
          ))}
        </nav>
        {currentDataset && (
          <div className="flex items-center gap-2 text-xs text-gray-400 ml-2">
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            <span className="truncate max-w-[200px]">{currentDataset.name}</span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-3">
          {pipelineRunning && (
            <div className="flex items-center gap-1.5 text-xs text-green-400">
              <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg>
              {pipelineStep}
            </div>
          )}
          {!pipelineRunning && pipelineStep && (
            <div className="text-xs text-green-400">{pipelineStep}</div>
          )}
        </div>
      </header>

      {/* Page content */}
      {page === 'upload' && (
        <UploadPage
          datasets={datasets}
          currentDataset={currentDataset}
          onSelectDataset={setCurrentDataset}
          onDatasetsChange={setDatasets}
          onNavigateTuning={() => navigateTo('tuning')}
          pipelineRunning={pipelineRunning}
          pipelineDatasetId={pipelineDatasetId}
          pipelineStep={pipelineStep}
        />
      )}
      {page === 'tuning' && (
        <TuningPage
          currentDataset={currentDataset}
          onNavigateResults={() => navigateTo('results')}
          onNavigateUpload={() => navigateTo('upload')}
          pipelineRunning={pipelineRunning}
          pipelineStep={pipelineStep}
          onRunPipeline={handleRunPipeline}
        />
      )}
      {page === 'results' && (
        <ResultsPage
          currentDataset={currentDataset}
          onNavigateTuning={() => navigateTo('tuning')}
        />
      )}
    </div>
  )
}
