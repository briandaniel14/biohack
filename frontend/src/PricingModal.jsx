import { useState } from 'react'

const FREE_FEATURES = [
  'Max 50 image files',
  'Max 5 concurrent runs',
  'Standard processing speed',
  '1 GB image storage',
  'Basic support',
  'No image customisation',
]

const PRO_FEATURES = [
  'Unlimited image files',
  'Unlimited concurrent processing runs',
  'Faster processing speed',
  'Unlimited file storage',
  'Priority support',
  'Priority image customisation',
]

export default function PricingModal({ open, onClose }) {
  const [loading, setLoading] = useState(false)

  if (!open) return null

  const handleUpgrade = async () => {
    setLoading(true)
    try {
      const resp = await fetch('/api/checkout', { method: 'POST' })
      const { url, error } = await resp.json()
      if (error) { alert(error); return }
      window.location.href = url
    } catch (e) {
      alert('Failed to start checkout: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl max-w-3xl w-full mx-4 overflow-hidden">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-gray-500 hover:text-gray-300 hover:bg-gray-800 transition-colors z-10"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Header */}
        <div className="px-8 pt-8 pb-4 text-center">
          <h2 className="text-2xl font-bold text-white">Choose Your Plan</h2>
          <p className="text-sm text-gray-400 mt-1">Unlock the full power of Filament Detection</p>
        </div>

        {/* Plans */}
        <div className="px-8 pb-8 grid grid-cols-2 gap-4">
          {/* Free */}
          <div className="rounded-xl border border-gray-800 bg-gray-800/40 p-6 flex flex-col">
            <h3 className="text-lg font-semibold text-gray-200">Free</h3>
            <div className="mt-2 mb-5">
              <span className="text-3xl font-bold text-white">£0</span>
              <span className="text-sm text-gray-500 ml-1">forever</span>
            </div>
            <ul className="space-y-2.5 flex-1">
              {FREE_FEATURES.map(f => (
                <li key={f} className="flex items-start gap-2 text-sm text-gray-400">
                  <svg className="w-4 h-4 mt-0.5 text-gray-600 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            <button
              disabled
              className="mt-6 w-full py-2.5 rounded-lg border border-gray-700 bg-gray-800 text-sm font-medium text-gray-500 cursor-default"
            >
              Current Plan
            </button>
          </div>

          {/* Pro */}
          <div className="rounded-xl border-2 border-blue-500/50 bg-gradient-to-b from-blue-900/20 to-gray-900 p-6 flex flex-col relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 bg-blue-600 rounded-full text-[11px] font-semibold text-white uppercase tracking-wide">
              Recommended
            </div>
            <h3 className="text-lg font-semibold text-white">Professional</h3>
            <div className="mt-2 mb-5">
              <span className="text-3xl font-bold text-white">£5</span>
              <span className="text-sm text-gray-400 ml-1">one-off payment</span>
            </div>
            <ul className="space-y-2.5 flex-1">
              {PRO_FEATURES.map(f => (
                <li key={f} className="flex items-start gap-2 text-sm text-gray-200">
                  <svg className="w-4 h-4 mt-0.5 text-blue-400 flex-none" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            <button
              onClick={handleUpgrade}
              disabled={loading}
              className="mt-6 w-full py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-semibold text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <><svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/></svg> Redirecting...</>
              ) : (
                'Upgrade Now'
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
