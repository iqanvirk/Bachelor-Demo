export type ScanCase = {
  id: string
  date: string
  time: string
  modality: string
  topImage: string | null
  bottomImage: string | null
  resultImage: string | null
  predictionImage: string | null
  findings: string[] | null
  confidence: number | null
}

export const mockScans: ScanCase[] = [
  {
    id: 'scan-1',
    date: '13.jan.2023',
    time: '08:53:30',
    modality: 'MTA',
    topImage: null,
    bottomImage: null,
    resultImage: null,
    predictionImage: null,
    findings: null,
    confidence: null,
  },
  {
    id: 'scan-2',
    date: '06.jun.2023',
    time: '13:15:35',
    modality: 'MTA',
    topImage: null,
    bottomImage: null,
    resultImage: null,
    predictionImage: null,
    findings: null,
    confidence: null,
  },
  {
    id: 'scan-3',
    date: '21.jan.2025',
    time: '14:10:53',
    modality: 'MTA',
    topImage: null,
    bottomImage: null,
    resultImage: null,
    predictionImage: null,
    findings: null,
    confidence: null,
  },
]

export const octLibrary: { id: string; src: string; label: string }[] = [
  { id: 'lib-1',  src: '/assets/img_115.png.webp',   label: 'img_115' },
  { id: 'lib-2',  src: '/assets/img_116.png.webp',   label: 'img_116' },
  { id: 'lib-3',  src: '/assets/img_118.png.webp',   label: 'img_118' },
  { id: 'lib-4',  src: '/assets/img_121.png.webp',   label: 'img_121' },
  { id: 'lib-5',  src: '/assets/img_124.png-2.webp', label: 'img_124' },
  { id: 'lib-6',  src: '/assets/img_124.png.webp',   label: 'img_124b' },
  { id: 'lib-7',  src: '/assets/img_125.png-2.webp', label: 'img_125' },
  { id: 'lib-8',  src: '/assets/img_125.png.webp',   label: 'img_125b' },
  { id: 'lib-9',  src: '/assets/img_126.png-2.webp', label: 'img_126' },
  { id: 'lib-10', src: '/assets/img_126.png.webp',   label: 'img_126b' },
]