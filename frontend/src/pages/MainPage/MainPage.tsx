import { useEffect, useState } from 'react';
import { Header } from '../../widgets/Header/Header';
import { HomeSection } from '../../widgets/HomeSection/HomeSection';
import { NewReportForm } from '../../widgets/NewReportForm/NewReportForm';
import { ListOfReports } from '../../widgets/ListOfReports/ListOfReports';
import type { Report } from '../../entities/report/model/types';
import {
  getMyReports,
  getReportStatus,
} from '../../shared/api/reportApi';
import cls from './MainPage.module.scss';

const isProcessingStatus = (status: string) => {
  return status.toLowerCase() === 'processing';
};

export const MainPage = () => {
  const [reports, setReports] = useState<Report[]>([]);

  useEffect(() => {
    const loadReports = async () => {
      try {
        const reportsFromServer = await getMyReports();
        setReports(reportsFromServer);
      } catch (error) {
        console.error('Не удалось загрузить список отчётов', error);
      }
    };

    loadReports();
  }, []);

  useEffect(() => {
    const processingReports = reports.filter((report) =>
      isProcessingStatus(report.status)
    );

    if (processingReports.length === 0) {
      return;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const updatedReports = await Promise.all(
          processingReports.map((report) => getReportStatus(report.id))
        );

        setReports((prevReports) =>
          prevReports.map((report) => {
            const updatedReport = updatedReports.find(
              (item) => item.id === report.id
            );

            if (!updatedReport) {
              return report;
            }

            return {
              ...report,
              ...updatedReport,
              patientName:
                updatedReport.patientName === 'Без имени'
                  ? report.patientName
                  : updatedReport.patientName,
              studyDate:
                updatedReport.studyDate === '—'
                  ? report.studyDate
                  : updatedReport.studyDate,
            };
          })
        );
      } catch (error) {
        console.error('Не удалось обновить статусы отчётов', error);
      }
    }, 5000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [reports]);

  const handleReportCreated = (report: Report) => {
    setReports((prevReports) => [report, ...prevReports]);
  };

  return (
    <div className={cls.page}>
      <Header />
      <HomeSection />

      <NewReportForm onReportCreated={handleReportCreated} />

      <Header />
      <ListOfReports reports={reports} />
    </div>
  );
};