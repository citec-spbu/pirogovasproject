import { memo, useState } from 'react';
import type { ChangeEvent, InputHTMLAttributes, MouseEvent } from 'react';
import eyeIcon from '../../assets/icons/eyeIcon.svg';
import eyeOffIcon from '../../assets/icons/eyeCloseIcon.svg';
import cls from './Input.module.scss';

type HTMLInputProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  'value' | 'onChange'
>;

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
  const isPassword = type === 'password';
  const shouldShowPasswordToggle = isPassword && showPasswordToggle;

  const actualType =
    isPassword && isPasswordVisible ? 'text' : type;

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange?.(event.target.value);
  };

  const togglePasswordVisibility = (
    event: MouseEvent<HTMLButtonElement>
  ) => {
    event.preventDefault();
    event.stopPropagation();

    setIsPasswordVisible((prev) => !prev);
  };

  return (
    <div className={cls.wrapper}>
      <label
        className={[
          cls.field,
          cls[variant],
          error ? cls.error : '',
          shouldShowPasswordToggle ? cls.withPasswordToggle : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
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

        {shouldShowPasswordToggle && (
          <button
            type="button"
            className={cls.eyeBtn}
            onClick={togglePasswordVisibility}
            aria-label={
              isPasswordVisible ? 'Скрыть пароль' : 'Показать пароль'
            }
          >
            <img
              src={isPasswordVisible ? eyeOffIcon : eyeIcon}
              alt=""
              className={cls.eyeIcon}
            />
          </button>
        )}
      </label>

      {error && <div className={cls.errorText}>{error}</div>}
    </div>
  );
});