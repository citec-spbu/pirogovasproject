import cls from './AdminDashboardPage.module.scss';

export const AdminDashboardPage = () => {
  return (
    <div className={cls.page}>
      <h1 className={cls.title}>Панель администратора</h1>

      <section className={cls.section}>
        <h2>Главная страница админки</h2>
      </section>
    </div>
  );
};