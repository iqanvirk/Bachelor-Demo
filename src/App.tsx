import { useMemo, useState } from 'react'
import { mockScans } from './data/mockScans'

type Screen = 'welcome' | 'image' | 'info'

function App() {
  const [screen, setScreen] = useState<Screen>('welcome')

  const selectValue = useMemo(() => {
    if (screen === 'image') return 'Bilde'
    return 'Informasjon'
  }, [screen])

  const handleSelectChange = (value: string) => {
    if (value === 'Bilde') setScreen('image')
    if (value === 'Informasjon') setScreen('info')
  }

  return (
    <div className="windows-shell">
      <img className="windows-topbar" src="/assets/windowstopbar.png" alt="Windows top bar" />

      <div className="app-shell">
        {screen === 'welcome' ? (
          <section className="welcome-screen">
            <div className="welcome-card">
              <h1>Demo av modellens implementasjon</h1>
              <p>Bachelorprosjekt Vår 2026</p>
              <button onClick={() => setScreen('image')}>Start</button>
            </div>
          </section>
        ) : (
          <main className="workspace">
            <aside className="left-rail">
              <div className="rail-icon active">AI</div>
            </aside>

            <section className="content-area">
              <div className="viewer-grid">
                {mockScans.map((scan) => (
                  <article key={scan.id} className="viewer-card">
                    <div className="viewer-toolbar">
                      <span>{scan.date}</span>
                      <span>{scan.time}</span>
                      <span>{scan.modality}</span>
                    </div>

                    <div className="fundus-frame">
                      <img src={scan.topImage} alt={`Retinavisning ${scan.date}`} />
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
                      <img src={scan.bottomImage} alt={`OCT-visning ${scan.date}`} />
                    </div>
                  </article>
                ))}
              </div>

              <section className="bottom-panel">
                <div className="bottom-panel-header">
                  <select
                    value={selectValue}
                    onChange={(e) => handleSelectChange(e.target.value)}
                    className="mode-select"
                  >
                    <option>Informasjon</option>
                    <option>Bilde</option>
                  </select>
                </div>

                {screen === 'info' ? (
                  <div className="info-grid">
                    {mockScans.map((scan) => (
                      <article key={scan.id} className="info-card">
                        <h3>
                          {scan.date} - {scan.time}
                        </h3>
                        <ul>
                          {scan.findings.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="result-grid">
                    {mockScans.map((scan) => (
                      <article key={scan.id} className="result-card">
                        <h3>
                          {scan.date} - {scan.time}
                        </h3>
                        <div className="result-image-frame">
                          <img src={scan.resultImage} alt={`Resultatbilde ${scan.date}`} />
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>
            </section>
          </main>
        )}
      </div>

      <img
        className="windows-bottombar"
        src="/assets/windowsbottombar.png"
        alt="Windows bottom bar"
      />
    </div>
  )
}

export default App