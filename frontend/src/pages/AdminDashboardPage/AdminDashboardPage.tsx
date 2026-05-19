import { useEffect, useState } from 'react';
import starFullIcon from '../../shared/assets/icons/starFullIcon.svg';
import { getAdminMetrics } from '../../shared/api/adminApi';
import cls from './AdminDashboardPage.module.scss';

type DashboardStats = {
  score: string;
  errors: string;
  responseTime: string;
};

export const AdminDashboardPage = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    getAdminMetrics()
      .then((data) => {
        setStats({
          score: data.average_review_score?.toFixed(1) ?? '—',
          errors: `${data.llm_error_percent}%`,
          responseTime: '—',
        });
      })
      .catch(() => {
        setStats({
          score: '—',
          errors: '—',
          responseTime: '—',
        });
      });
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
