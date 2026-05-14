import { Header } from '../../widgets/Header/Header';
import { HomeSection } from '../../widgets/HomeSection/HomeSection';
import { NewReportForm } from '../../widgets/NewReportForm/NewReportForm';
import { ListOfReports } from '../../widgets/ListOfReports/ListOfReports';
import cls from './MainPage.module.scss';

export const MainPage = () => {
  return (
    <div className={cls.page}>
      <Header />
      <HomeSection />
      <NewReportForm/>
      <Header />
      <ListOfReports/>

    </div>
  );
};