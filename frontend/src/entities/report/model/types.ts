export type ReportStatus = 'PROCESSING' | 'READY' | 'ERROR' | 'processing' | 'ready' | 'error';

export interface Report {
  id: string;
  patientName: string;
  studyDate: string;
  status: ReportStatus;
  htmlReady?: boolean;
  pdfReady?: boolean;
  errorMessage?: string | null;
  reviewScore?: number | null;
  reviewText?: string | null;
}