import cls from './ListOfReports.module.scss';
import EyeIcon from '../../shared/assets/icons/eyeIcon.svg';
import DownloadIcon from '../../shared/assets/icons/downloadIcon.svg';
import type { Report } from '../../entities/report/model/types';
import { ReportReview } from '../../features/report-review/ReportReview';
import {
  addReportReview,
  openPdfReport,
  viewHtmlReport,
} from '../../shared/api/reportApi';

interface ListOfReportsProps {
  reports: Report[];
}

const normalizeStatus = (status: string) => status.toLowerCase();

const getStatusLabel = (status: string) => {
  const normalizedStatus = normalizeStatus(status);

  if (normalizedStatus === 'processing') {
    return 'Формируется';
  }

  if (normalizedStatus === 'ready' || normalizedStatus === 'completed') {
    return 'Готов';
  }

  if (normalizedStatus === 'error' || normalizedStatus === 'failed') {
    return 'Ошибка';
  }

  return status;
};

const isReportReady = (report: Report) => {
  const normalizedStatus = normalizeStatus(report.status);

  return (
    normalizedStatus === 'ready' ||
    normalizedStatus === 'completed' ||
    report.htmlReady ||
    report.pdfReady
  );
};

export const ListOfReports = ({ reports }: ListOfReportsProps) => {
  const handleViewClick = async (reportId: string) => {
    try {
      await viewHtmlReport(reportId);
    } catch (error) {
      console.error('Не удалось открыть отчёт', error);
    }
  };

  const handleDownloadClick = async (reportId: string) => {
    try {
      await openPdfReport(reportId);
    } catch (error) {
      console.error('Не удалось скачать отчёт', error);
    }
  };

  const handleReviewSubmit = async (
    reportId: string,
    rating: number,
    comment: string
  ) => {
    await addReportReview(reportId, rating, comment);
  };

  return (
    <section className={cls.wrapper}>
      <div className={cls.table}>
        <div id="reports-title" className={cls.anchor} />

        <div className={cls.header}>
          <div className={cls.cell}>№ отчёта</div>
          <div className={cls.cell}>Имя пациента</div>
          <div className={cls.cell}>Дата исследования</div>
          <div className={cls.cell}>Статус</div>
          <div className={cls.cell}>Просмотр</div>
          <div className={cls.cell}>Скачать</div>
          <div className={cls.cell}>Оценка</div>
        </div>

        {reports.length === 0 && (
          <div className={cls.empty}>Отчётов пока нет</div>
        )}

        {reports.map((report, index) => {
          const ready = isReportReady(report);

          return (
            <div className={cls.row} key={report.id}>
              <div className={cls.cell}>{index + 1}</div>
              <div className={cls.cell}>{report.patientName}</div>
              <div className={cls.cell}>{report.studyDate}</div>
              <div className={cls.cell}>{getStatusLabel(report.status)}</div>

              <div className={cls.cell}>
                <button
                  className={cls.iconButton}
                  type="button"
                  disabled={!ready}
                  onClick={() => handleViewClick(report.id)}
                >
                  <img src={EyeIcon} alt="Просмотр отчёта" />
                </button>
              </div>

              <div className={cls.cell}>
                <button
                  className={cls.iconButton}
                  type="button"
                  disabled={!ready}
                  onClick={() => handleDownloadClick(report.id)}
                >
                  <img src={DownloadIcon} alt="Скачать отчёт"  />
                </button>
              </div>

              <div className={cls.cell}>
                <ReportReview
                  disabled={!ready}
                  initialRating={report.reviewScore}
                  initialComment={report.reviewText}
                  onSubmit={(rating, comment) =>
                    handleReviewSubmit(report.id, rating, comment)
                  }
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};