import { useState } from 'react'
import { mockScans } from './data/mockScans'

type Screen = 'welcome' | 'image' | 'info'

export default function App() {
  const [screen, setScreen] = useState<Screen>('welcome')

  return (
    <div className="windows-shell">
      <img
        className="windows-topbar"
        src="/assets/windowstopbar.png"
        alt="Windows topp"
      />

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
            {/* Left rail — AI-knapp nederst */}
            <div className="left-rail">
              <div className="rail-icon">AI</div>
            </div>

            <div className="content-area">
              {/* Tre viewer-kolonner */}
              <div className="viewer-grid">
                {mockScans.map((scan) => (
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
                        <span className="tool-toggle">
                          <span className="tool-dot" /> OD
                        </span>
                        <span className="tool-close">×</span>
                      </div>
                    </div>

                    <div className="fundus-frame">
                      <img src={scan.topImage} alt={`Retina ${scan.date}`} />
                    </div>

                    <div className="oct-scale">
                      <span>0</span>
                      <span>100</span>
                      <span>200</span>
                      <span>300</span>
                      <span>400</span>
                      <span>500 µm</span>
                    </div>

                    <div className="oct-frame">
                      <img src={scan.bottomImage} alt={`OCT ${scan.date}`} />
                    </div>
                  </div>
                ))}
              </div>

              {/* Bunnpanel */}
              <div className="bottom-panel">
                <div className="bottom-panel-header">
                  <div className="mode-select-wrap">
                    <span className="mode-dot" />
                    <select
                      className="mode-select"
                      value={screen === 'info' ? 'Informasjon' : 'Bilde'}
                      onChange={(e) =>
                        setScreen(e.target.value === 'Informasjon' ? 'info' : 'image')
                      }
                    >
                      <option>Bilde</option>
                      <option>Informasjon</option>
                    </select>
                  </div>
                </div>

                {screen === 'info' ? (
                  <div className="info-grid">
                    {mockScans.map((scan) => (
                      <div key={scan.id} className="info-card">
                        <h3>{scan.date} - {scan.time}</h3>
                        <ul>
                          {scan.findings.map((item, i) => (
                            <li key={i}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="result-grid">
                    {mockScans.map((scan) => (
                      <div key={scan.id} className="result-card">
                        <h3>{scan.date} - {scan.time}</h3>
                        <div className="result-image-frame">
                          <img src={scan.resultImage} alt={`Resultat ${scan.date}`} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <img
        className="windows-bottombar"
        src="/assets/windowsbottombar.png"
        alt="Windows bunn"
      />
    </div>
  )
}