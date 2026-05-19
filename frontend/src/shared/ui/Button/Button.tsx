import type { ButtonHTMLAttributes, ReactNode } from 'react';
import cls from './Button.module.scss';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  className?: string;
}

export const Button = ({ children, className = '', ...props }: ButtonProps) => {
  return (
    <button className={`${cls.button} ${className}`} {...props}>
      {children}
    </button>
  );
};