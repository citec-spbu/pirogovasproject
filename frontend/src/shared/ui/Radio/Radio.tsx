import cls from './Radio.module.scss';

interface RadioOption {
  value: string;
  label: string;
}

interface RadioProps {
  label: string;
  name: string;
  value: string;
  options: RadioOption[];
  onChange: (value: string) => void;
  error?: string;
}

export const Radio= ({
  label,
  name,
  value,
  options,
  onChange,
  error,
}: RadioProps) => {
  return (
    <div className={cls.wrapper}>
      <p className={cls.label}>{label}</p>

      <div className={cls.options}>
        {options.map((option) => (
          <label key={option.value} className={cls.option}>
            <input
              className={cls.radio}
              type="radio"
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />

            <span className={cls.text}>{option.label}</span>
          </label>
        ))}
      </div>

      {error && <p className={cls.errorText}>{error}</p>}
    </div>
  );
};