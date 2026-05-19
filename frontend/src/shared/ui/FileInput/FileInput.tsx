import type { ChangeEvent, DragEvent } from 'react';
import { useRef, useState } from 'react';
import FileIcon from '../../assets/icons/fileIcon.svg';
import cls from './FileInput.module.scss';

interface FileInputProps {
  label?: string;
  files: File[];
  onChange: (files: File[]) => void;

  allowedExtensions: string[];
  multiple?: boolean;
  required?: boolean;

  error?: string;
  placeholder?: string;
}

export const FileInput = ({
  label,
  files,
  onChange,
  allowedExtensions,
  multiple = true,
  required = false,
  error,
  placeholder,
}: FileInputProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [localError, setLocalError] = useState('');
  const [isDragActive, setIsDragActive] = useState(false);

  const normalizedExtensions = allowedExtensions.map((ext) =>
    ext.toLowerCase().replace('.', '')
  );

  const acceptValue = normalizedExtensions.map((ext) => `.${ext}`).join(',');

  const getFileExtension = (fileName: string) => {
    return fileName.split('.').pop()?.toLowerCase() ?? '';
  };

  const validateFile = (file: File) => {
    const extension = getFileExtension(file.name);

    if (!normalizedExtensions.includes(extension)) {
      return `Файл "${file.name}" имеет недопустимый формат. Разрешены: ${normalizedExtensions
        .map((ext) => `.${ext}`)
        .join(', ')}`;
    }

    return '';
  };

  const isDuplicate = (newFile: File, currentFiles: File[]) => {
    return currentFiles.some(
      (file) =>
        file.name === newFile.name &&
        file.size === newFile.size &&
        file.lastModified === newFile.lastModified
    );
  };

  const addFiles = (selectedFiles: File[]) => {
    setLocalError('');

    if (selectedFiles.length === 0) {
      if (required) {
        setLocalError('Поле обязательно для заполнения');
      }

      return;
    }

    const validFiles: File[] = [];

    for (const file of selectedFiles) {
      const validationError = validateFile(file);

      if (validationError) {
        setLocalError(validationError);
        continue;
      }

      if (!isDuplicate(file, files)) {
        validFiles.push(file);
      }
    }

    if (validFiles.length === 0) {
      return;
    }

    const updatedFiles = multiple
      ? [...files, ...validFiles]
      : validFiles.slice(0, 1);

    onChange(updatedFiles);
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files ?? []);

    addFiles(selectedFiles);

    event.target.value = '';
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setIsDragActive(false);

    const droppedFiles = Array.from(event.dataTransfer.files);
    addFiles(droppedFiles);
  };

  const removeFile = (indexToRemove: number) => {
    const updatedFiles = files.filter((_, index) => index !== indexToRemove);

    onChange(updatedFiles);

    if (required && updatedFiles.length === 0) {
      setLocalError('Поле обязательно для заполнения');
    } else {
      setLocalError('');
    }
  };

  const clearFiles = () => {
    onChange([]);

    if (inputRef.current) {
      inputRef.current.value = '';
    }

    if (required) {
      setLocalError('Поле обязательно для заполнения');
    } else {
      setLocalError('');
    }
  };

  const currentError = error || localError;

  const extensionsText = normalizedExtensions
    .map((ext) => `.${ext}`)
    .join(', ');

  const mainText = placeholder || `Выберите файл (${extensionsText})`;

  return (
    <div className={cls.wrapper}>
      {label && (
        <p className={cls.label}>
          {label}
          {required && <span className={cls.required}> *</span>}
        </p>
      )}

      <label
        className={[
          cls.fileButton,
          isDragActive ? cls.dragActive : '',
          currentError ? cls.error : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className={cls.fileInput}
          type="file"
          accept={acceptValue}
          multiple={multiple}
          onChange={handleInputChange}
        />

        <img className={cls.fileIcon} src={FileIcon} alt="" />

        <span className={cls.text}>
          {mainText}
          <span className={cls.dragText}> или перетащите сюда</span>
        </span>
      </label>

      {files.length > 0 && (
        <div className={cls.fileList}>
          <div className={cls.fileListHeader}>
            <span className={cls.fileCount}>
              Загружено файлов: {files.length}
            </span>

            <button
              type="button"
              className={cls.clearButton}
              onClick={clearFiles}
            >
              Очистить всё
            </button>
          </div>

          {files.map((file, index) => (
            <div
              className={cls.fileItem}
              key={`${file.name}-${file.size}-${file.lastModified}`}
            >
              <div className={cls.fileInfo}>
                <span className={cls.fileName}>{file.name}</span>
                <span className={cls.fileSize}>
                  {(file.size / 1024 / 1024).toFixed(2)} МБ
                </span>
              </div>

              <button
                type="button"
                className={cls.removeButton}
                onClick={() => removeFile(index)}
              >
                Удалить
              </button>
            </div>
          ))}
        </div>
      )}

      {currentError && <p className={cls.errorText}>{currentError}</p>}
    </div>
  );
};