import { memo, useState } from 'react';
import type { ChangeEvent, InputHTMLAttributes } from 'react';
import cls from './Input.module.scss';

type HTMLInputProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>;

type InputVariant = 'default' | 'floating';

interface InputProps extends HTMLInputProps {
  value?: string;
  onChange?: (value: string) => void;
  label?: string;
  error?: string;
  variant?: InputVariant;
  showPasswordToggle?: boolean;
}

export const Input = memo((props: InputProps) => {
  const {
    value = '',
    onChange,
    type = 'text',
    label,
    error,
    variant = 'default',
    showPasswordToggle = true,
    ...otherProps
  } = props;

  const [isPasswordVisible, setIsPasswordVisible] = useState(false);

  const isFloating = variant === 'floating';

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    onChange?.(e.target.value);
  };

  const togglePasswordVisibility = () => {
    setIsPasswordVisible((prev) => !prev);
  };

  const actualType =
    type === 'password' ? (isPasswordVisible ? 'text' : 'password') : type;

  return (
    <div className={cls.wrapper}>
      <label className={`${cls.field} ${cls[variant]} ${error ? cls.error : ''}`}>
        {isFloating ? (
          <>
            <input
              className={cls.input}
              type={actualType}
              value={value}
              onChange={handleChange}
              placeholder=" "
              {...otherProps}
            />
            {label && <span className={cls.floatingLabel}>{label}</span>}
          </>
        ) : (
          <>
            {label && <span className={cls.defaultLabel}>{label}</span>}
            <input
              className={cls.input}
              type={actualType}
              value={value}
              onChange={handleChange}
              {...otherProps}
            />
          </>
        )}

        {/* {type === 'password' && showPasswordToggle && (
          <button
            type="button"
            className={cls.eyeBtn}
            onClick={togglePasswordVisibility}
          >
            {isPasswordVisible ? '🙈' : '👁'}
          </button>
        )} */}
        
      </label>

      {error && <div className={cls.errorText}>{error}</div>}
    </div>
  );
});