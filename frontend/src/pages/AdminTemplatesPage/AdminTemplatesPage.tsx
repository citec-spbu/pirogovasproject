import { useState } from 'react';
import { FileInput } from '../../shared/ui/FileInput/FileInput';
import { Button } from '../../shared/ui/Button/Button';
import downloadIcon from '../../shared/assets/icons/downloadIcon.svg';
import cls from './AdminTemplatesPage.module.scss';

type TemplateVersion = {
  id: string;
  version: string;
  status: 'active' | 'inactive';
  date: string;
};

const mockVersions: TemplateVersion[] = [
  {
    id: '1',
    version: 'random',
    status: 'active',
    date: 'XX.XX.XXXX',
  },
];

const getCurrentDate = () => {
  return new Date().toLocaleDateString('ru-RU');
};

export const AdminTemplatesPage = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [versions, setVersions] = useState<TemplateVersion[]>(mockVersions);

  const handleUpload = () => {
    if (files.length === 0) {
      console.log('Файл шаблона не выбран');
      return;
    }

    const file = files[0];

    const newVersion: TemplateVersion = {
      id: crypto.randomUUID(),
      version: file.name,
      status: 'inactive',
      date: getCurrentDate(),
    };

    setVersions((prevVersions) => [newVersion, ...prevVersions]);
    setFiles([]);

    console.log('Загружен шаблон:', file);
  };

  return (
    <div className={cls.page}>
      <section className={cls.uploadSection}>
        <h1 className={cls.title}>Загрузка нового шаблона</h1>

        <FileInput
          files={files}
          onChange={setFiles}
          allowedExtensions={['html']}
          multiple={false}
          placeholder="Выберите файл (.html)"
        />

        <Button
          type="button"
          className={cls.uploadButton}
          onClick={handleUpload}
        >
          Загрузить
        </Button>
      </section>

      <section className={cls.historySection}>
        <h2 className={cls.subtitle}>История версий</h2>

        <table className={cls.table}>
          <thead>
            <tr>
              <th>Версия</th>
              <th>Статус</th>
              <th>Дата</th>
              <th>Действия</th>
            </tr>
          </thead>

          <tbody>
            {versions.map((version) => (
              <tr key={version.id}>
                <td>{version.version}</td>
                <td>{version.status}</td>
                <td>{version.date}</td>
                <td>
                  <button
                    type="button"
                    className={cls.iconButton}
                    aria-label="Скачать шаблон"
                  >
                    <img src={downloadIcon} alt="" className={cls.downloadIcon}/>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
};