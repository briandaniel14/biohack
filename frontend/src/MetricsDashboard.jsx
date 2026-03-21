import { useMemo } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from 'recharts'
import { getTrackColor, PX_TO_UM } from './data.js'

const TT = {
  background: '#1f2937', border: 'none', fontSize: 10, padding: '4px 8px', borderRadius: 4
}

export default function MetricsDashboard({ rows, filamentSummary, onJumpToFrame, filteredFilamentIds }) {
  const handleChartClick = (data) => {
    if (data && data.activeLabel != null && onJumpToFrame) {
      onJumpToFrame(data.activeLabel)
    }
  }
  const frames = useMemo(() => {
    if (!rows.length) return []
    const maxF = Math.max(...rows.map(r => r.frame))
    return Array.from({ length: maxF + 1 }, (_, i) => i)
  }, [rows])

  const filamentIds = useMemo(() => {
    const all = filamentSummary.map(f => f.filament_ID)
    if (filteredFilamentIds == null) return all
    return all.filter(id => filteredFilamentIds.includes(id))
  }, [filamentSummary, filteredFilamentIds])

  const xTicks = useMemo(() => {
    if (!frames.length) return []
    const max = frames[frames.length - 1]
    const step = max <= 20 ? 1 : max <= 50 ? 5 : 10
    const ticks = []
    for (let i = 0; i <= max; i += step) ticks.push(i)
    return ticks
  }, [frames])

  // Chart 1: Filament length over time (per filament_ID)
  const lengthData = useMemo(() => {
    return frames.map(f => {
      const row = { frame: f }
      rows.filter(r => r.frame === f && r.filament_present === 1 && r.filament_ID != null).forEach(r => {
        row[`f${r.filament_ID}`] = +(r.filament_mean_length_px * PX_TO_UM).toFixed(2)
      })
      return row
    })
  }, [rows, frames])

  const maxLengths = useMemo(() => {
    return filamentIds.map(fid => {
      const vals = rows
        .filter(r => r.filament_ID === fid && r.filament_mean_length_px != null)
        .map(r => r.filament_mean_length_px)
      return { fid, max: vals.length > 0 ? +(Math.max(...vals) * PX_TO_UM).toFixed(2) : 0 }
    }).filter(v => v.max > 0)
  }, [rows, filamentIds])

  // Chart 2:
  const countData = useMemo(() => {
    return frames.map(f => {
      const frameRows = rows.filter(r => r.frame === f)
      const filamentRows = frameRows.filter(r => r.filament_present === 1 && (filteredFilamentIds == null || filteredFilamentIds.includes(r.filament_ID)))
      return {
        frame: f,
        filaments: filamentRows.length,
        cells: frameRows.length,
      }
    })
  }, [rows, frames, filteredFilamentIds])

  // Chart 3: Filament area over time (per filament_ID)
  const areaData = useMemo(() => {
    return frames.map(f => {
      const row = { frame: f }
      rows.filter(r => r.frame === f && r.filament_present === 1 && r.filament_ID != null).forEach(r => {
        row[`f${r.filament_ID}`] = +(r.filament_area * PX_TO_UM * PX_TO_UM).toFixed(2)
      })
      return row
    })
  }, [rows, frames])

  // Chart 4: Filament eccentricity over time (per filament_ID)
  const eccData = useMemo(() => {
    return frames.map(f => {
      const row = { frame: f }
      rows.filter(r => r.frame === f && r.filament_present === 1 && r.filament_ID != null).forEach(r => {
        row[`f${r.filament_ID}`] = r.filament_eccentricity
      })
      return row
    })
  }, [rows, frames])

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 p-2 h-full">
      {/* Chart 1: Filament length */}
      <ChartPanel title="Filament Length Over Time">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={lengthData} onClick={handleChartClick} style={{cursor:'pointer'}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} ticks={xTicks} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} label={{ value: 'µm', angle: -90, position: 'insideLeft', style: { fontSize: 9, fill: '#9ca3af' } }} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {filamentIds.map(fid => (
              <Line key={fid} type="monotone" dataKey={`f${fid}`} stroke={getTrackColor(fid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`Filament ${Math.round(fid)}`} />
            ))}
            {maxLengths.map(({ fid, max }) => (
              <ReferenceLine key={`max-${fid}`} y={max} stroke={getTrackColor(fid)} strokeDasharray="4 2" strokeWidth={1} strokeOpacity={0.5} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 2: Filaments per frame */}
      <ChartPanel title="Filaments Per Frame">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={countData} barGap={0} onClick={handleChartClick} style={{cursor:'pointer'}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} ticks={xTicks} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} allowDecimals={false} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            <Bar dataKey="filaments" fill="#60a5fa" name="Filaments" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 3: Filament area */}
      <ChartPanel title="Filament Area Over Time">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={areaData} onClick={handleChartClick} style={{cursor:'pointer'}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} ticks={xTicks} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} label={{ value: 'µm²', angle: -90, position: 'insideLeft', style: { fontSize: 9, fill: '#9ca3af' } }} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {filamentIds.map(fid => (
              <Line key={fid} type="monotone" dataKey={`f${fid}`} stroke={getTrackColor(fid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`Filament ${Math.round(fid)}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      {/* Chart 4: Filament eccentricity */}
      <ChartPanel title="Filament Eccentricity Over Time">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={eccData} onClick={handleChartClick} style={{cursor:'pointer'}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="frame" tick={{ fontSize: 9, fill: '#9ca3af' }} ticks={xTicks} />
            <YAxis tick={{ fontSize: 9, fill: '#9ca3af' }} domain={[0, 1]} />
            <Tooltip contentStyle={TT} labelFormatter={v => `Frame ${v}`} />
            {filamentIds.map(fid => (
              <Line key={fid} type="monotone" dataKey={`f${fid}`} stroke={getTrackColor(fid)} dot={false} strokeWidth={1.5} connectNulls={false} name={`Filament ${Math.round(fid)}`} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>
    </div>
  )
}

function ChartPanel({ title, children }) {
  return (
    <div className="bg-gray-900 rounded p-2 flex flex-col min-h-0">
      <div className="text-[10px] text-white font-semibold uppercase mb-1 flex-none">{title}</div>
      <div className="flex-1 min-h-0">{children}</div>
    </div>
  )
}
