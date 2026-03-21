import { useState, useEffect, useRef, useCallback } from 'react'
import { loadDatasets, loadDatasetSummary, runPipeline, pollJobStatus } from './data.js'
import UploadPage from './UploadPage.jsx'
import TuningPage from './TuningPage.jsx'
import ResultsPage from './ResultsPage.jsx'
import PricingModal from './PricingModal.jsx'

const PAGES = [
  { key: 'upload', label: 'Upload', icon: 'M12 16V4m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4' },
  { key: 'tuning', label: 'Tuning', icon: 'M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4' },
  { key: 'results', label: 'Results', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
]

export default function App() {
  const [page, setPage] = useState('upload')
  const [datasets, setDatasets] = useState([])
  const [currentDataset, setCurrentDataset] = useState(null)
  const [showPricing, setShowPricing] = useState(false)

  // Pipeline jobs — supports concurrent runs, persists across page refresh
  const [pipelineJobs, setPipelineJobs] = useState(() => {
    try { const s = sessionStorage.getItem('pipelineJobs'); return s ? JSON.parse(s) : {} }
    catch { return {} }
  })
  const pollRefs = useRef({})
  useEffect(() => { sessionStorage.setItem('pipelineJobs', JSON.stringify(pipelineJobs)) }, [pipelineJobs])

  // Derived compat state
  const pipelineRunning = Object.values(pipelineJobs).some(j => j.running)
  const activeJob = Object.values(pipelineJobs).find(j => j.running) || Object.values(pipelineJobs).find(j => j.step)
  const pipelineStep = activeJob?.step || ''
  const pipelineDatasetId = activeJob?.datasetId || null

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

  // Resume polling on mount (survives refresh), cleanup on unmount
  useEffect(() => {
    Object.entries(pipelineJobs).forEach(([jid, job]) => {
      if (job.running && !pollRefs.current[jid]) {
        pollRefs.current[jid] = setInterval(async () => {
          try {
            const st = await pollJobStatus(jid)
            if (st.status === 'complete') {
              clearInterval(pollRefs.current[jid]); delete pollRefs.current[jid]
              setPipelineJobs(p => { const n = {...p}; n[jid] = {...n[jid], running: false, step: 'Done!'}; return n })
              loadDatasets().then(ds => setDatasets(ds)).catch(() => {})
              setTimeout(() => setPipelineJobs(p => { const n = {...p}; delete n[jid]; return n }), 5000)
            } else if (st.status === 'error') {
              clearInterval(pollRefs.current[jid]); delete pollRefs.current[jid]
              setPipelineJobs(p => { const n = {...p}; n[jid] = {...n[jid], running: false, step: 'Error: ' + (st.error||'Failed')}; return n })
            } else {
              setPipelineJobs(p => { const n = {...p}; n[jid] = {...n[jid], step: st.step || 'Processing...'}; return n })
            }
          } catch {}
        }, 2000)
      }
    })
    return () => { Object.values(pollRefs.current).forEach(id => clearInterval(id)) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunPipeline = useCallback(async (datasetId, params) => {
    const alreadyRunning = Object.values(pipelineJobs).some(j => j.running && j.datasetId === datasetId)
    if (alreadyRunning) return
    try {
      const { job_id, error } = await runPipeline(datasetId, params)
      if (error) { setPipelineJobs(p => ({...p, ['e'+Date.now()]: {datasetId, running: false, step: 'Error: '+error}})); return }
      setPipelineJobs(p => ({...p, [job_id]: {datasetId, running: true, step: 'Starting...'}}))
      pollRefs.current[job_id] = setInterval(async () => {
        try {
          const st = await pollJobStatus(job_id)
          if (st.status === 'complete') {
            clearInterval(pollRefs.current[job_id]); delete pollRefs.current[job_id]
            setPipelineJobs(p => { const n = {...p}; n[job_id] = {...n[job_id], running: false, step: 'Done!'}; return n })
            loadDatasets().then(ds => setDatasets(ds)).catch(() => {})
            setTimeout(() => setPipelineJobs(p => { const n = {...p}; delete n[job_id]; return n }), 5000)
          } else if (st.status === 'error') {
            clearInterval(pollRefs.current[job_id]); delete pollRefs.current[job_id]
            setPipelineJobs(p => { const n = {...p}; n[job_id] = {...n[job_id], running: false, step: 'Error: '+(st.error||'Failed')}; return n })
          } else {
            setPipelineJobs(p => { const n = {...p}; n[job_id] = {...n[job_id], step: st.step || 'Processing...'}; return n })
          }
        } catch {}
      }, 2000)
    } catch (e) {
      setPipelineJobs(p => ({...p, ['e'+Date.now()]: {datasetId, running: false, step: 'Error: '+e.message}}))
    }
  }, [pipelineJobs])

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
        <div className="ml-auto">
          <button
            onClick={() => setShowPricing(true)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-700 hover:bg-blue-600 text-white text-sm font-medium shadow-sm shadow-blue-700/25 transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Upgrade
          </button>
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
          onNavigateResults={() => navigateTo('results')}
          pipelineJobs={pipelineJobs}
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
          pipelineRunning={Object.values(pipelineJobs).some(j => j.running && j.datasetId === currentDataset?.id)}
          pipelineStep={(() => { const j = Object.values(pipelineJobs).find(j => j.datasetId === currentDataset?.id); return j?.step || '' })()}
          onRunPipeline={handleRunPipeline}
        />
      )}
      {page === 'results' && (
        <ResultsPage
          currentDataset={currentDataset}
          onNavigateTuning={() => navigateTo('tuning')}
        />
      )}
      <PricingModal open={showPricing} onClose={() => setShowPricing(false)} />
    </div>
  )
}
