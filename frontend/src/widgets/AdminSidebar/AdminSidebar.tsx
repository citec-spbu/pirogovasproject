import { NavLink } from 'react-router-dom';
import cls from './AdminSidebar.module.scss';

export const AdminSidebar = () => {
  return (
    <aside className={cls.sidebar}>
      <nav className={cls.nav}>
        <NavLink to="/admin" end className={cls.link}>
          Главная
        </NavLink>

        <NavLink to="/admin/templates" className={cls.link}>
          Шаблоны
        </NavLink>

        <NavLink to="/admin/users" className={cls.link}>
          Пользователи
        </NavLink>

        <NavLink to="/admin/protocols" className={cls.link}>
          Протоколы
        </NavLink>
      </nav>
    </aside>
  );
};