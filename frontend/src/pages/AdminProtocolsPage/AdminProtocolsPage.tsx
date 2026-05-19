import { useMemo, useState } from 'react';
import { SearchBar } from '../../shared/ui/SearchBar/SearchBar';
import { FileInput } from '../../shared/ui/FileInput/FileInput';
import { Button } from '../../shared/ui/Button/Button';
import fileIcon from '../../shared/assets/icons/fileBlueIcon.svg';
import cls from './AdminProtocolsPage.module.scss';

const mockProtocols = [
  '158-157-1-PB.pdf',
  '337-649-1-SM.pdf',
  'Аортальная недостаточность.pdf',
  'Аортальный стеноз.pdf',
  'Расслоение аорты_final.pdf',
  'Расслоение аорты.pdf',
  'Рекомендации брюшная аорта.pdf',
  'Рекомендации_торакоабдоминальная_аорта.pdf',
  'abdominal-aortic-aneurysm.pdf',
  'etz2015.pdf',
  'isselbacher_et_al_2022_2022_acc_aha_guideline_for_the_diagnosis.pdf',
  'jtd-09-05-S551.pdf',
  'Kardiologiya_2018_01_007.pdf',
  'Recom po aorte 7_rkj_15.pdf',
  'recommendation.pdf',
  'recommendation2.pdf',
  'rogers2013.pdf',
  'ziganshin2019.pdf'
];

export const AdminProtocolsPage = () => {
  const [searchValue, setSearchValue] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [protocols, setProtocols] = useState<string[]>(mockProtocols);

  const filteredProtocols = useMemo(() => {
    const normalizedSearch = searchValue.trim().toLowerCase();

    if (!normalizedSearch) {
      return protocols;
    }

    return protocols.filter((protocol) =>
      protocol.toLowerCase().includes(normalizedSearch)
    );
  }, [searchValue, protocols]);

  const handleSearchSubmit = () => {
    console.log('Поиск протокола:', searchValue);
  };

  const handleUpload = () => {
    if (files.length === 0) {
      console.log('Файл протокола не выбран');
      return;
    }

    const uploadedFileNames = files.map((file) => file.name);

    setProtocols((prevProtocols) => [
      ...uploadedFileNames,
      ...prevProtocols,
    ]);

    setFiles([]);

    console.log('Загружены протоколы:', files);
  };

  return (
    <div className={cls.page}>
      <SearchBar
        value={searchValue}
        onChange={setSearchValue}
        onSearch={handleSearchSubmit}
        placeholder="Поиск..."
      />

      <section className={cls.uploadSection}>
        <h1 className={cls.title}>Загрузка нового протокола</h1>

        <FileInput
          files={files}
          onChange={setFiles}
          allowedExtensions={['zip', 'rar', '7z']}
          multiple
          placeholder="Выберите файл (.zip, .rar, .7z)"
        />

        <Button type="button" className={cls.uploadButton} onClick={handleUpload}>
          Загрузить
        </Button>
      </section>

      <section className={cls.currentSection}>
        <h2 className={cls.subtitle}>Текущие клинические рекомендации</h2>

        <ul className={cls.fileList}>
          {filteredProtocols.map((protocol, index) => (
            <li key={`${protocol}-${index}`} className={cls.fileItem}>
              <img src={fileIcon} alt="" className={cls.protocolIcon} />
              <span>{protocol}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
};