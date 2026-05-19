import type { ReactNode, MouseEvent } from 'react';
import cls from './Modal.module.scss';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}

export const Modal = ({ isOpen, onClose, children }: ModalProps) => {
  if (!isOpen) return null;

  const handleContentClick = (e: MouseEvent<HTMLDivElement>) => {
    e.stopPropagation();
  };

  return (
    <div className={cls.overlay} onClick={onClose}>
      <div className={cls.content} onClick={handleContentClick}>
        {children}
      </div>
    </div>
  );
};