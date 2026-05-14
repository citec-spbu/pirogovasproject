import type { ChangeEvent } from 'react';
import FileIcon from '../../assets/icons/fileIcon.svg';
import cls from './FileInput.module.scss';


interface FileInputProps {
  label: string;
  file: File | null;
  onChange: (file: File | null) => void;
  error?: string;
}

export const FileInput = ({ label, file, onChange, error }: FileInputProps) => {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] ?? null;
    onChange(selectedFile);
  };

  return (
    <div className={cls.wrapper}>
      <p className={cls.label}>{label}</p>

      <label className={`${cls.fileButton} ${error ? cls.error : ''}`}>
        <input
          className={cls.fileInput}
          type="file"
          onChange={handleChange}
        />
        
        <img className={cls.FileIcon} src={FileIcon} alt="File" />

        <span className={cls.text}>
          {file ? file.name : 'Выберите файл'}
        </span>
      </label>

      {error && <p className={cls.errorText}>{error}</p>}
    </div>
  );
};