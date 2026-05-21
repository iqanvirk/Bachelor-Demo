import { useState } from 'react'
import type { ScanCase } from './data/mockScans'
import { mockScans, octLibrary } from './data/mockScans'
import ImageLibrarySidebar from './components/ImageLibrarySidebar'

type Screen = 'welcome' | 'image' | 'info' | 'AI-analyse'

async function runModel(imageSrc: string): Promise<{
  predictionImage: string | null
  findings: string[]
  confidence: number | null
  finalPrediction: string | null
  imagePrediction: string | null
  thicknessPrediction: string | null
}> {
  const imageResponse = await fetch(imageSrc)
  const imageBlob = await imageResponse.blob()
  const formData = new FormData()
  formData.append('file', imageBlob, 'oct-image.png')

  const response = await fetch('http://127.0.0.1:8000/analyze', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('Analysis failed')
  }

  const data = await response.json()

  return {
    predictionImage: data.overlay_base64
      ? `data:image/png;base64,${data.overlay_base64}`
      : null,
    confidence: typeof data.confidence === 'number' ? data.confidence : null,
    finalPrediction: data.final_prediction ?? null,
    imagePrediction: data.image_prediction ?? null,
    thicknessPrediction: data.thickness_prediction ?? null,
    findings: [
      `CNN-modell prediksjon: ${data.image_prediction ?? '—'}`,
      `Tykkelsesmodell prediksjon: ${data.thickness_prediction ?? '—'}`,
      `Endelig prediksjon: ${data.final_prediction ?? '—'}`,
      `Konfidens: ${typeof data.confidence === 'number' ? `${(data.confidence * 100).toFixed(1)}%` : '—'}`,
      `Patologisk score: ${typeof data.pathology_score === 'number' ? data.pathology_score.toFixed(2) : '—'}`,
      `Gjennomsnittlig retinal tykkelse: ${
        typeof data.total_retinal_metrics?.mean_total_retinal_thickness_px === 'number'
          ? `${data.total_retinal_metrics.mean_total_retinal_thickness_px.toFixed(1)} px`
          : '—'
      }`,
    ],
  }
}

function getPredictionStatus(scan: ScanCase) {
  const prediction = scan.finalPrediction?.toLowerCase()

  if (prediction === 'healthy') {
    return {
      label: 'Prediction: Healthy',
      className: 'prediction-status healthy',
    }
  }

  if (prediction === 'unhealthy') {
    return {
      label: 'Prediction: Unhealthy',
      className: 'prediction-status unhealthy',
    }
  }

  if (prediction === 'uncertain') {
    return {
      label: 'Prediction: Uncertain',
      className: 'prediction-status uncertain',
    }
  }

  return null
}

const IconBarChart = () => (
  <svg viewBox="0 0 14 14" xmlns="http://www.w3.org/2000/svg">
    <rect x="1" y="8" width="3" height="5" />
    <rect x="5.5" y="5" width="3" height="8" />
    <rect x="10" y="2" width="3" height="11" />
  </svg>
)

const IconLayers = () => (
  <svg viewBox="0 0 14 14" xmlns="http://www.w3.org/2000/svg">
    <polygon points="7,1 13,4.5 7,8 1,4.5" />
    <polyline points="1,7 7,10.5 13,7" />
    <polyline points="1,10 7,13.5 13,10" />
  </svg>
)

export default function App() {
  const [screen, setScreen] = useState<Screen>('welcome')
  const [scans, setScans] = useState<ScanCase[]>(mockScans)
  const [analysing, setAnalysing] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarSlot, setSidebarSlot] = useState<number | null>(null)

  const handleLibrarySelect = (src: string) => {
    if (sidebarSlot === null) return
    setScans((prev) =>
      prev.map((s, i) =>
        i === sidebarSlot ? { ...s, bottomImage: src, topImage: src } : s
      )
    )
    setSidebarOpen(false)
    setSidebarSlot(null)
  }

  const handleRunAnalysis = async () => {
    setAnalysing(true)
    const updated = [...scans]
    await Promise.all(
      updated.map(async (scan, i) => {
        if (!scan.bottomImage) return
        const result = await runModel(scan.bottomImage)
        updated[i] = {
          ...updated[i],
          predictionImage: result.predictionImage,
          resultImage: result.predictionImage,
          findings: result.findings,
          confidence: result.confidence,
          finalPrediction: result.finalPrediction,
          imagePrediction: result.imagePrediction,
          thicknessPrediction: result.thicknessPrediction,
        }
      })
    )
    setScans(updated)
    setAnalysing(false)
    setScreen('image')
  }

  const anyHasImage = scans.some((s) => s.bottomImage !== null)

  return (
    <div className="windows-shell">
      <img className="windows-topbar" src="/assets/windowstopbar.png" alt="Windows topp" />

      <div className="app-shell">
        {screen === 'welcome' ? (
          <div className="welcome-screen">
            <div className="welcome-card">
              <h1>Demo av modellens implementasjon</h1>
              <p>Bachelorprosjekt Vår 2026</p>
              <button onClick={() => setScreen('image')}>Start</button>
            </div>
          </div>
        ) : (
          <div className="workspace">
            <div className="left-rail">
              <div className="rail-icon-passive"><IconBarChart /></div>
              <div className="rail-icon-passive"><IconLayers /></div>
              <div className="rail-icon">AI</div>
            </div>

            <div className="content-area">
              <div className="viewer-grid">
                {scans.map((scan, i) => (
                  <div key={scan.id} className="viewer-card">
                    <div className="viewer-toolbar">
                      <div className="viewer-toolbar-left">
                        <span>{scan.date}</span>
                        <span>{scan.time}</span>
                        <span>{scan.modality}</span>
                      </div>
                      <div className="viewer-toolbar-right">
                        <span className="tool-icon">◌</span>
                        <span className="tool-icon">☆</span>
                        <span className="tool-toggle"><span className="tool-dot" /> OD</span>
                        <span className="tool-close">×</span>
                      </div>
                    </div>

                    <div className="fundus-frame">
                      {scan.topImage
                        ? <img src={scan.topImage} alt={`Retina ${scan.date}`} />
                        : <span className="no-data">No data available yet.</span>}
                    </div>

                    <div className="oct-scale">
                      <span>0</span><span>100</span><span>200</span>
                      <span>300</span><span>400</span><span>500 µm</span>
                    </div>

                    <div className="oct-frame">
                      {scan.bottomImage
                        ? <img src={scan.bottomImage} alt={`OCT ${scan.date}`} />
                        : <span className="no-data">No data available yet.</span>}
                    </div>

                    {screen === 'AI-analyse' && (
                      <div className="upload-row">
                        <button
                          className="upload-btn"
                          onClick={() => { setSidebarSlot(i); setSidebarOpen(true) }}
                          title="Åpne bildebibliotek"
                        >
                          <svg viewBox="0 0 12 12" xmlns="http://www.w3.org/2000/svg">
                            <path d="M1 8v2a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V8" />
                            <polyline points="4,4 6,2 8,4" />
                            <line x1="6" y1="2" x2="6" y2="8" />
                          </svg>
                          Last opp
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {screen === 'AI-analyse' && (
                <div className="run-bar">
                  <button
                    className={`run-btn${!anyHasImage ? ' disabled' : ''}${analysing ? ' running' : ''}`}
                    onClick={handleRunAnalysis}
                    disabled={!anyHasImage || analysing}
                  >
                    {analysing ? 'Analyserer...' : 'Kjør analyse'}
                  </button>
                </div>
              )}

              <div className="bottom-panel">
                <div className="bottom-panel-header">
                  <div className="mode-select-wrap">
                    <span className="mode-dot" />
                    <select
                      className="mode-select"
                      value={
                        screen === 'info' ? 'Informasjon'
                        : screen === 'AI-analyse' ? 'AI-analyse'
                        : 'Bilde'
                      }
                      onChange={(e) => {
                        const v = e.target.value
                        if (v === 'Bilde') setScreen('image')
                        else if (v === 'Informasjon') setScreen('info')
                        else setScreen('AI-analyse')
                      }}
                    >
                      <option>Bilde</option>
                      <option>Informasjon</option>
                      <option>AI-analyse</option>
                    </select>
                  </div>
                </div>

                {screen === 'info' && (
                  <div className="info-grid">
                    {scans.map((scan) => (
                      <div key={scan.id} className="info-card">
                        <h3>{scan.date} – {scan.time}</h3>
                        {scan.findings
                          ? <ul>{scan.findings.map((item, idx) => <li key={idx}>{item}</li>)}</ul>
                          : <span className="no-data">No data available yet.</span>}
                      </div>
                    ))}
                  </div>
                )}

                {(screen === 'image' || screen === 'AI-analyse') && (
                  <div className="result-grid">
                    {scans.map((scan) => {
                      const status = getPredictionStatus(scan)

                      return (
                        <div key={scan.id} className="result-card">
                          <h3>{scan.date} – {scan.time}</h3>

                          {status && (
                            <div className={status.className}>
                              {status.label}
                            </div>
                          )}

                          <div className="result-image-frame">
                            {scan.predictionImage
                              ? <img src={scan.predictionImage} alt={`Resultat ${scan.date}`} />
                              : <span className="no-data">No data available yet.</span>}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>

            <div className="right-rail" />
          </div>
        )}
      </div>

      <img className="windows-bottombar" src="/assets/windowsbottombar.png" alt="Windows bunn" />

      <ImageLibrarySidebar
        open={sidebarOpen}
        onClose={() => { setSidebarOpen(false); setSidebarSlot(null) }}
        selectedId={
          sidebarSlot !== null
            ? (octLibrary.find(img => img.src === scans[sidebarSlot]?.bottomImage)?.id ?? '')
            : ''
        }
        onSelect={(id) => {
          const found = octLibrary.find(img => img.id === id)
          if (found) handleLibrarySelect(found.src)
        }}
      />
    </div>
  )
}