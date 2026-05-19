import { useEffect, useState } from 'react';
import starFullIcon from '../../shared/assets/icons/starFullIcon.svg';
import cls from './AdminDashboardPage.module.scss';

type DashboardStats = {
  score: string;
  errors: string;
  responseTime: string;
};

const mockDashboardStats: DashboardStats = {
  score: '4.8',
  errors: '2.3%',
  responseTime: '30 с.',
};

export const AdminDashboardPage = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    // Потом здесь будет запрос к backend:
    // const data = await getAdminDashboardStats();
    // setStats(data);

    setStats(mockDashboardStats);
  }, []);

  if (!stats) {
    return (
      <div className={cls.page}>
        <p className={cls.loading}>Загрузка статистики...</p>
      </div>
    );
  }

  const dashboardStats = [
    {
      label: 'Оценка',
      value: stats.score,
      icon: starFullIcon,
    },
    {
      label: 'Ошибки',
      value: stats.errors,
    },
    {
      label: 'Время ответа',
      value: stats.responseTime,
    },
  ];

  return (
    <div className={cls.page}>
      <div className={cls.stats}>
        {dashboardStats.map((stat) => (
          <div key={stat.label} className={cls.statCard}>
            <p className={cls.statLabel}>{stat.label}</p>

            <div className={cls.valueRow}>
              <p className={cls.statValue}>{stat.value}</p>

              {stat.icon && (
                <img src={stat.icon} alt="" className={cls.starIcon} />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};