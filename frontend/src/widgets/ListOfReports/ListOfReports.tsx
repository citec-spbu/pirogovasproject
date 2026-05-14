import { useState } from 'react';
import cls from './ListOfReports.module.scss';
import EyeIcon from '../../shared/assets/icons/eyeIcon.svg';
import DownloadIcon from '../../shared/assets/icons/downloadIcon.svg';

interface Report {
  id: number;
  patientName: string;
  studyDate: string;
  status: string;
}

const reports: Report[] = [
  {
    id: 1,
    patientName: 'Иванов И. И.',
    studyDate: '12.05.2026',
    status: 'Готов',
  },
];

export const ListOfReports = () => {
  const [ratings, setRatings] = useState<Record<number, number>>({});

  const handleRatingClick = (reportId: number, rating: number) => {
    setRatings((prev) => ({
      ...prev,
      [reportId]: rating,
    }));
  };

  return (
    <section className={cls.wrapper}>
      <div className={cls.table}>
        <div className={cls.header}>
          <div className={cls.cell}>№ отчёта</div>
          <div className={cls.cell}>Имя пациента</div>
          <div className={cls.cell}>Дата исследования</div>
          <div className={cls.cell}>Статус</div>
          <div className={cls.cell}>Просмотр</div>
          <div className={cls.cell}>Скачать</div>
          <div className={cls.cell}>Оценка</div>
        </div>

        {reports.map((report) => {
          const currentRating = ratings[report.id] ?? 0;

          return (
            <div className={cls.row} key={report.id}>
              <div className={cls.cell}>{report.id}</div>
              <div className={cls.cell}>{report.patientName}</div>
              <div className={cls.cell}>{report.studyDate}</div>
              <div className={cls.cell}>{report.status}</div>

              <div className={cls.cell}>
                <button className={cls.iconButton} type="button">
                  <img src={EyeIcon} alt="Просмотр отчёта" />
                </button>
              </div>

              <div className={cls.cell}>
                <button className={cls.iconButton} type="button">
                  <img src={DownloadIcon} alt="Скачать отчёт" />
                </button>
              </div>

              <div className={cls.cell}>
                <div className={cls.rating}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      className={`${cls.starButton} ${
                        star <= currentRating ? cls.activeStar : ''
                      }`}
                      onClick={() => handleRatingClick(report.id, star)}
                      aria-label={`Оценить на ${star}`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};