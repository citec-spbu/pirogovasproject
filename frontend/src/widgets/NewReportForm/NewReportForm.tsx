import { useState } from 'react';
import type { FormEvent } from 'react';
import { Input } from '../../shared/ui/Input/Input';
import { Button } from '../../shared/ui/Button/Button';
import { FileInput } from '../../shared/ui/FileInput/FileInput';
import { Radio } from '../../shared/ui/Radio/Radio';
import type { Report } from '../../entities/report/model/types';
import { createReport } from '../../shared/api/reportApi';
import cls from './NewReportForm.module.scss';

interface NewReportFormProps {
  onReportCreated: (report: Report) => void;
}

export const NewReportForm = ({ onReportCreated }: NewReportFormProps) => {
  const [patientName, setPatientName] = useState('');
  const [gender, setGender] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [ctDate, setCtDate] = useState('');
  const [anamnesis, setAnamnesis] = useState('');

  const [ctFiles, setCtFiles] = useState<File[]>([]);
  const [measurementsFiles, setMeasurementsFiles] = useState<File[]>([]);

  const [patientNameError, setPatientNameError] = useState('');
  const [genderError, setGenderError] = useState('');
  const [birthDateError, setBirthDateError] = useState('');
  const [ctDateError, setCtDateError] = useState('');
  const [ctFilesError, setCtFilesError] = useState('');
  const [measurementsFilesError, setMeasurementsFilesError] = useState('');
  const [submitError, setSubmitError] = useState('');

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    let hasError = false;

    setPatientNameError('');
    setGenderError('');
    setBirthDateError('');
    setCtDateError('');
    setCtFilesError('');
    setMeasurementsFilesError('');
    setSubmitError('');

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

    if (ctFiles.length === 0) {
      setCtFilesError('Загрузите архив с КТ-снимками');
      hasError = true;
    }

    if (measurementsFiles.length === 0) {
      setMeasurementsFilesError('Загрузите файл измерений');
      hasError = true;
    }

    if (hasError) {
      return;
    }

    try {
      setIsSubmitting(true);

      const createdReport = await createReport({
        patientName,
        gender,
        birthDate,
        ctDate,
        anamnesis,
        ctImages: ctFiles[0],
        measurementsFile: measurementsFiles[0],
      });

      onReportCreated(createdReport);

      setPatientName('');
      setGender('');
      setBirthDate('');
      setCtDate('');
      setAnamnesis('');
      setCtFiles([]);
      setMeasurementsFiles([]);
    } catch (error) {
      if (error instanceof Error) {
        setSubmitError(error.message);
      } else {
        setSubmitError('Не удалось сформировать отчёт');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className={cls.page}>
      <div className={cls.formWrapper}>
      <h1 id="new-report-title" className={cls.title}>Новый отчёт</h1>

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
              { value: 'Female', label: 'Женский' },
              { value: 'Male', label: 'Мужской' },
            ]}
          />

          <Input
            label="Дата рождения пациента *"
            value={birthDate}
            onChange={setBirthDate}
            variant="default"
            placeholder="ДД.ММ.ГГГГ"
            error={birthDateError}
          />

          <Input
            label="Дата КТ-исследования *"
            value={ctDate}
            onChange={setCtDate}
            variant="default"
            placeholder="ДД.ММ.ГГГГ"
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
            label="Загрузите КТ-снимки"
            files={ctFiles}
            onChange={(files) => {
              setCtFiles(files);

              if (files.length > 0) {
                setCtFilesError('');
              }
            }}
            allowedExtensions={['zip']}
            multiple={false}
            required
            error={ctFilesError}
            placeholder="Выберите архив (.zip)"
          />

          <FileInput
            label="Загрузите измерения"
            files={measurementsFiles}
            onChange={(files) => {
              setMeasurementsFiles(files);

              if (files.length > 0) {
                setMeasurementsFilesError('');
              }
            }}
            allowedExtensions={['json', 'csv']}
            multiple={false}
            required
            error={measurementsFilesError}
            placeholder="Выберите файл (.json, .csv)"
          />

          {submitError && <p className={cls.errorText}>{submitError}</p>}

          <Button type="submit" className={cls.submitButton}>
            {isSubmitting ? 'Формируется...' : 'Сформировать отчёт'}
          </Button>
        </form>
      </div>
    </section>
  );
};