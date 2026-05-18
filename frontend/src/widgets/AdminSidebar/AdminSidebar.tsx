import { NavLink } from 'react-router-dom';
import cls from './AdminSidebar.module.scss';

const adminNavItems = [
  {
    to: '/admin',
    label: 'Home Dashboard',
    end: true,
  },
  {
    to: '/admin/templates',
    label: 'Шаблоны',
  },
  {
    to: '/admin/users',
    label: 'Пользователи',
  },
  {
    to: '/admin/protocols',
    label: 'Протоколы',
  },
];

export const AdminSidebar = () => {
  return (
    <aside className={cls.sidebar}>
      <nav className={cls.nav}>
        {adminNavItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              [cls.link, isActive ? cls.active : ''].filter(Boolean).join(' ')
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
};