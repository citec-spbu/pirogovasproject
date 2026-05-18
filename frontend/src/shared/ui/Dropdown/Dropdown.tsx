import type { ReactNode } from 'react';
import { useEffect, useRef, useState } from 'react';
import cls from './Dropdown.module.scss';

type DropdownChildren = ReactNode | ((close: () => void) => ReactNode);

type DropdownProps = {
  trigger: ReactNode;
  children: DropdownChildren;
  className?: string;
  triggerClassName?: string;
  menuClassName?: string;
  align?: 'left' | 'right';
};

export const Dropdown = ({
  trigger,
  children,
  className,
  triggerClassName,
  menuClassName,
  align = 'right',
}: DropdownProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const close = () => {
    setIsOpen(false);
  };

  const toggle = () => {
    setIsOpen((prev) => !prev);
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handleClickOutside = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        close();
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        close();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  return (
    <div ref={rootRef} className={[cls.dropdown, className].filter(Boolean).join(' ')}>
      <button
        type="button"
        className={[cls.trigger, triggerClassName].filter(Boolean).join(' ')}
        onClick={toggle}
        aria-haspopup="menu"
        aria-expanded={isOpen}
      >
        {trigger}
      </button>

      {isOpen && (
        <div
          className={[
            cls.menu,
            align === 'right' ? cls.right : cls.left,
            menuClassName,
          ]
            .filter(Boolean)
            .join(' ')}
          role="menu"
        >
          {typeof children === 'function' ? children(close) : children}
        </div>
      )}
    </div>
  );
};