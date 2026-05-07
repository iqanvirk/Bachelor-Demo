export type ScanCase = {
  id: string
  date: string
  time: string
  modality: string
  topImage: string
  bottomImage: string
  resultImage: string
  findings: string[]
  confidence: number
}

export const mockScans: ScanCase[] = [
  {
    id: 'scan-1',
    date: '13.jan.2023',
    time: '08:53:30',
    modality: 'MTA',
    topImage: '/assets/retina3d.webp',
    bottomImage: '/assets/octscan.png',
    resultImage: '/assets/resultat-bilde.webp',
    findings: [
      'Gjennomsnittlig retinal tykkelse: 298 µm',
      'Sentral makulatykkelse (CMT): 265 µm',
      'Intraretinal væske: Estimert volum 0,51 mm³',
      'Retinal lagstruktur: Milde uregelmessigheter',
      'AI-vurdering: Avvik påvist',
      'Modellkonfidens: 93%',
    ],
    confidence: 93,
  },
  {
    id: 'scan-2',
    date: '06.jun.2023',
    time: '13:15:35',
    modality: 'MTA',
    topImage: '/assets/retina3d.webp',
    bottomImage: '/assets/octscan.png',
    resultImage: '/assets/resultat-bilde.webp',
    findings: [
      'Gjennomsnittlig retinal tykkelse: 312 µm (+4,7 %)',
      'Sentral makulatykkelse (CMT): 278 µm (+4,9 %)',
      'Intraretinal væske: Estimert volum 0,42 mm³ (-17,6 %)',
      'Retinal lagstruktur: Milde uregelmessigheter',
      'AI-vurdering: Reduksjon',
      'Modellkonfidens: 94%',
    ],
    confidence: 94,
  },
  {
    id: 'scan-3',
    date: '21.jan.2025',
    time: '14:10:53',
    modality: 'MTA',
    topImage: '/assets/retina3d.webp',
    bottomImage: '/assets/octscan.png',
    resultImage: '/assets/resultat-bilde.webp',
    findings: [
      'Gjennomsnittlig retinal tykkelse: 305 µm (-2,2 %)',
      'Sentral makulatykkelse (CMT): 271 µm (-2,5 %)',
      'Intraretinal væske: Estimert volum 0,39 mm³ (-7,1 %)',
      'Retinal lagstruktur: Milde uregelmessigheter',
      'AI-vurdering: Reduksjon',
      'Modellkonfidens: 95%',
    ],
    confidence: 95,
  },
]