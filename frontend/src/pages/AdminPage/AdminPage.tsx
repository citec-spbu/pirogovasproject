import cls from './AdminPage.module.scss';

export const AdminPage = () => {
  return (
    <main className={cls.page}>
      <h1 className={cls.title}>Панель администратора</h1>

      <section className={cls.section}>
        <h2>Пользователи</h2>
        <p>Здесь будет создание и редактирование пользователей.</p>
      </section>

      <section className={cls.section}>
        <h2>Шаблоны отчётов</h2>
        <p>Здесь будет загрузка шаблонов отчётов.</p>
      </section>

      <section className={cls.section}>
        <h2>Клинические протоколы</h2>
        <p>Здесь будет загрузка клинических протоколов.</p>
      </section>
    </main>
  );
};