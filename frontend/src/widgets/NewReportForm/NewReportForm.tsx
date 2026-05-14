import { useState } from 'react';
import type { FormEvent } from 'react';
import { Input } from '../../shared/ui/Input/Input';
import { Button } from '../../shared/ui/Button/Button';
import { FileInput } from '../../shared/ui/FileInput/FileInput';
import { Radio } from '../../shared/ui/Radio/Radio';
import cls from './NewReportForm.module.scss';

export const NewReportForm = () => {
  const [patientName, setPatientName] = useState('');
  const [gender, setGender] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [ctDate, setCtDate] = useState('');
  const [anamnesis, setAnamnesis] = useState('');

  const [ctFile, setCtFile] = useState<File | null>(null);
  const [measurementsFile, setMeasurementsFile] = useState<File | null>(null);

  const [patientNameError, setPatientNameError] = useState('');
  const [genderError, setGenderError] = useState('');
  const [birthDateError, setBirthDateError] = useState('');
  const [ctDateError, setCtDateError] = useState('');
  const [ctFileError, setCtFileError] = useState('');
  const [measurementsFileError, setMeasurementsFileError] = useState('');

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    let hasError = false;

    setPatientNameError('');
    setGenderError('');
    setBirthDateError('');
    setCtDateError('');
    setCtFileError('');
    setMeasurementsFileError('');

    if (!patientName.trim()) {
      setPatientNameError('Введите ФИО пациента или шифр');
      hasError = true;
    }

    if (!gender) {
      setGenderError('Выберите пол пациента');
      hasError = true;
    }

    if (!birthDate.trim()) {
      setBirthDateError('Введите дату рождения пациента');
      hasError = true;
    }

    if (!ctDate.trim()) {
      setCtDateError('Введите дату КТ-исследования');
      hasError = true;
    }

    if (!ctFile) {
      setCtFileError('Загрузите КТ-снимки');
      hasError = true;
    }

    if (!measurementsFile) {
      setMeasurementsFileError('Загрузите измерения');
      hasError = true;
    }

    if (hasError) {
      return;
    }

    console.log({
      patientName,
      gender,
      birthDate,
      ctDate,
      anamnesis,
      ctFile,
      measurementsFile,
    });
  };

  return (
    <section className={cls.page}>
      <div className={cls.formWrapper}>
        <h1 className={cls.title}>Новый отчёт</h1>

        <form className={cls.form} onSubmit={handleSubmit}>
          <Input
            label="ФИО пациента / Шифр *"
            value={patientName}
            onChange={setPatientName}
            variant="default"
            placeholder="Иванов Иван Иванович / Шифр"
            error={patientNameError}
          />

          <Radio
            label="Пол пациента *"
            name="gender"
            value={gender}
            onChange={setGender}
            error={genderError}
            options={[
              { value: 'female', label: 'Женский' },
              { value: 'male', label: 'Мужской' },
            ]}
          />

          <Input
            label="Дата рождения пациент *"
            value={birthDate}
            onChange={setBirthDate}
            variant="default"
            placeholder="XX.XX.XXXX"
            error={birthDateError}
          />

          <Input
            label="Дата КТ-исследования *"
            value={ctDate}
            onChange={setCtDate}
            variant="default"
            placeholder="XX.XX.XXXX"
            error={ctDateError}
          />

          <Input
            label="Анамнез"
            value={anamnesis}
            onChange={setAnamnesis}
            variant="default"
            placeholder="Введите текст"
          />

          <div className={cls.divider} />

          <FileInput
            label="Загрузите КТ-снимки *"
            file={ctFile}
            onChange={setCtFile}
            error={ctFileError}
          />

          <FileInput
            label="Загрузите измерения *"
            file={measurementsFile}
            onChange={setMeasurementsFile}
            error={measurementsFileError}
          />

          <Button type="submit" className={cls.submitButton}>
            Сформировать отчёт
          </Button>
        </form>
      </div>

    </section>
  );
};