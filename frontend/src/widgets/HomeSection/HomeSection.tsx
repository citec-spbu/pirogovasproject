import cls from './HomeSection.module.scss';

export const HomeSection = () => {
  return (
    <section className={cls.hero}>
      <h1 className={cls.title}>Автоотчёты по КТ</h1>

      <p className={cls.subtitle}>
        Загрузите КТ-снимки и данные и получите отчёт.
      </p>

      <div className={cls.actions}>
        <button className={cls.secondary}>Просмотр отчётов</button>
        <button className={cls.secondary}>Новый отчёт</button>
      </div>
    </section>
  );
};