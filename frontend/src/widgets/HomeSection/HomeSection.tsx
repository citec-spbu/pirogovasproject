import cls from './HomeSection.module.scss';

export const HomeSection = () => {
  const scrollToElement = (id: string) => {
    const element = document.getElementById(id);

    if (!element) {
      console.error(`Элемент с id="${id}" не найден`);
      return;
    }

    element.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  };

  return (
    <section className={cls.hero}>
      <h1 className={cls.title}>Автоотчёты по КТ</h1>

      <p className={cls.subtitle}>
        Загрузите КТ-снимки и данные и получите отчёт.
      </p>

      <div className={cls.actions}>
        <button
          type="button"
          className={cls.secondary}
          onClick={() => scrollToElement('reports-title')}
        >
          Просмотр отчётов
        </button>

        <button
          type="button"
          className={cls.secondary}
          onClick={() => scrollToElement('new-report-title')}
        >
          Новый отчёт
        </button>
      </div>
    </section>
  );
};