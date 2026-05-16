import { X } from 'lucide-react'
import { octLibrary } from '../data/mockScans'
import clsx from 'clsx'

interface Props {
  open: boolean
  onClose: () => void
  selectedId: string
  onSelect: (id: string) => void
}

export default function ImageLibrarySidebar({ open, onClose, selectedId, onSelect }: Props) {
  return (
    <>
      {open && <div className="sidebar-backdrop" onClick={onClose} />}

      <div className={clsx('image-library-sidebar', open && 'open')}>
        <div className="sidebar-header">
          <span>BILDEBIBLIOTEK</span>
          <button onClick={onClose} aria-label="Lukk">
            <X size={16} />
          </button>
        </div>

        <div className="sidebar-grid">
          {octLibrary.map((img) => (
            <button
              key={img.id}
              className={clsx('lib-thumb', selectedId === img.id && 'active')}
              onClick={() => onSelect(img.id)}
            >
              <img
                src={img.src}
                alt={img.label}
                loading="lazy"
                width="100%"
                style={{ aspectRatio: '3/1', objectFit: 'cover', display: 'block', borderRadius: '5px' }}
              />
            </button>
          ))}
        </div>
      </div>
    </>
  )
}